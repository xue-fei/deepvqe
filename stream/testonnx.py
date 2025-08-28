import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from modules.convolution import StreamConv2d
from modules.convert import convert_to_stream


class FE(nn.Module):
    """Feature extraction"""
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c
    def forward(self, x):
        """x: (B,F,1,2)"""
        x_mag = torch.sqrt(x[...,[0]]**2 + x[...,[1]]**2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1-self.c) + 1e-12)  # (B,F,T,2)
        return x_c.permute(0,3,2,1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = StreamConv2d(channels, channels, kernel_size=(4,3), padding=(0,1))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()
    def forward(self, x, cache):
        """
        x: (B,C,1,F)
        cache: (B,C,3,F)
        """
        y, cache = self.conv(x, cache)
        y = self.elu(self.bn(y))
        return y + x, cache
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), stride=(1,2)):
        super().__init__()
        self.conv = StreamConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(0,1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.resblock = ResidualBlock(out_channels)
    def forward(self, x, conv_cache, res_cache):
        """
        x: (B,C,1,F)
        conv_cache: (B,Ci,3,Fi)
        res_cache:  (B,Co,3,Fo)
        """
        x, conv_cache = self.conv(x, conv_cache)
        x = self.elu(self.bn(x))
        x, res_cache = self.resblock(x, res_cache)
        return x, conv_cache, res_cache


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, cache):
        """x : (B,C,1,F)"""
        y = rearrange(x, 'b c t f -> b t (c f)')
        y, cache = self.gru(y, cache)
        y = self.fc(y)
        y = rearrange(y, 'b t (c f) -> b c t f', f=x.shape[-1])
        return y, cache
    

class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3)):
        super().__init__()
        self.conv = StreamConv2d(in_channels, out_channels*2, kernel_size, padding=(0,1))
        
    def forward(self, x, cache):
        """
        x: (B,C,1,F)
        cache: (B,C,3,F)
        """
        y, cache = self.conv(x, cache)
        y = rearrange(y, 'b (r c) t f -> b c t (r f)', r=2)
        return y, cache
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), is_last=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.is_last = is_last
    def forward(self, x, x_en, conv_cache, res_cache):
        """
        x: (B,C,1,F)
        x_en: (B,C,1,F)
        conv_cache: (B,Ci,2,Fi)
        res_cache: (B,Ci,2,Fi)
        """
        y = x + self.skip_conv(x_en)
        y, res_cache = self.resblock(y, res_cache)
        y, conv_cache = self.deconv(y, conv_cache)
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y, conv_cache, res_cache
    

class CCM(nn.Module):
    """Complex convolving mask block"""
    def __init__(self):
        super().__init__()
        # self.v = torch.tensor([1, -1/2 + 1j*np.sqrt(3)/2, -1/2 - 1j*np.sqrt(3)/2], dtype=torch.complex64)
        self.v = torch.tensor([[1,        -1/2,           -1/2],
                               [0, np.sqrt(3)/2, -np.sqrt(3)/2]], dtype=torch.float32)  # (2,3)
        self.unfold = nn.Unfold(kernel_size=(3,3), padding=(0,1))
    
    def forward(self, m, x, cache):
        """
        m: (B,27,1,F)
        x: (B,F,1,2)
        cache: (B,F,2,2)
        """
        m = m.view(1, 3, 9, 1, 257)
        H_real = torch.sum(self.v[0].to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)
        H_imag = torch.sum(self.v[1].to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)
        
        M_real = H_real.view(1, 3, 3, 1, 257)
        M_imag = H_imag.view(1, 3, 3, 1, 257)
        
        x = torch.cat([cache, x], dim=2)     # (B,F,T,2)
        cache = x[:,:,1:]                    # (B,F,2,2)
        x = x.permute(0,3,2,1).contiguous()  # (B,2,T,F)

        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(1, 2, 3, 3, 1, 257)
        
        x_enh_real = torch.sum(M_real * x_unfold[:,0] - M_imag * x_unfold[:,1], dim=(1,2))  # (B,T,F)
        x_enh_imag = torch.sum(M_real * x_unfold[:,1] + M_imag * x_unfold[:,0], dim=(1,2))  # (B,T,F)
        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3).transpose(1,2).contiguous()

        return x_enh, cache
        
        
        
    """ONNX模型"""
    import time
    import onnx
    import onnxruntime
    from onnxsim import simplify
    import soundfile as sf
    from scipy import signal  # 用于STFT计算
    
## run onnx model
    file = 'onnx_models/deepvqe_simple.onnx'
    # session = onnxruntime.InferenceSession(file, None, providers=['CPUExecutionProvider'])
    session = onnxruntime.InferenceSession(file, None, providers=['CPUExecutionProvider'])
    en_conv_cache1 = np.zeros([1,2,3,257],  dtype="float32")
    en_res_cache1  = np.zeros([1,64,3,129], dtype="float32")
    en_conv_cache2 = np.zeros([1,64,3,129], dtype="float32")
    en_res_cache2  = np.zeros([1,128,3,65], dtype="float32")
    en_conv_cache3 = np.zeros([1,128,3,65], dtype="float32")
    en_res_cache3  = np.zeros([1,128,3,33], dtype="float32")
    en_conv_cache4 = np.zeros([1,128,3,33], dtype="float32")
    en_res_cache4  = np.zeros([1,128,3,17], dtype="float32")
    en_conv_cache5 = np.zeros([1,128,3,17], dtype="float32")
    en_res_cache5  = np.zeros([1,128,3,9 ], dtype="float32")
    h_cache        = np.zeros([1,1,64*9  ], dtype="float32")
    de_res_cache5  = np.zeros([1,128,3,9 ], dtype="float32")
    de_conv_cache5 = np.zeros([1,128,3,9 ], dtype="float32")
    de_res_cache4  = np.zeros([1,128,3,17], dtype="float32")
    de_conv_cache4 = np.zeros([1,128,3,17], dtype="float32")
    de_res_cache3  = np.zeros([1,128,3,33], dtype="float32")
    de_conv_cache3 = np.zeros([1,128,3,33], dtype="float32")
    de_res_cache2  = np.zeros([1,128,3,65], dtype="float32")
    de_conv_cache2 = np.zeros([1,128,3,65], dtype="float32")
    de_res_cache1  = np.zeros([1,64,3,129], dtype="float32")
    de_conv_cache1 = np.zeros([1,64,3,129], dtype="float32")
    m_cache        = np.zeros([1,257,2,2],  dtype="float32")

    # 读取WAV文件
    audio_path = '../test.wav'  # 替换为您的WAV文件路径
    x, fs = sf.read(audio_path, dtype='float32')  # x: (n_samples,), 单声道, 16 kHz

    # 计算STFT
    n_fft = 512
    hop_length = 256
    f, t, Zxx = signal.stft(x, fs, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')

    # 将复数STFT结果转换为实部和虚部分开的格式
    stft_real = np.real(Zxx)
    stft_imag = np.imag(Zxx)

    # 组合为模型输入格式 (1, F, T, 2)
    stft_input = np.stack([stft_real, stft_imag], axis=-1)
    stft_input = np.expand_dims(stft_input, axis=0)  # 添加批次维度

    # 确保频率维度正确 (应该是257)
    if stft_input.shape[1] != 257:
        # 如果频率维度不正确，可能需要调整
        print(f"警告: STFT频率维度为{stft_input.shape[1]}，但期望为257")
        # 这里可以根据需要进行截断或填充

    T_list = []
    outputs = []

    # 处理每一帧
    for i in range(stft_input.shape[2]):  # 遍历时间维度
        tic = time.perf_counter()
        
        # 提取当前帧 (保持形状为 [1, 257, 1, 2])
        current_frame = stft_input[:, :, i:i+1, :]
        
        # 运行ONNX模型
        out_i, en_conv_cache1, en_res_cache1, en_conv_cache2, en_res_cache2, en_conv_cache3, en_res_cache3,\
                en_conv_cache4, en_res_cache4, en_conv_cache5, en_res_cache5,\
                h_cache, de_conv_cache5, de_res_cache5, de_conv_cache4, de_res_cache4, de_conv_cache3, de_res_cache3,\
                de_conv_cache2, de_res_cache2, de_conv_cache1, de_res_cache1, m_cache\
                = session.run([], {'mix': current_frame,
                    'en_conv_cache1': en_conv_cache1, 'en_res_cache1': en_res_cache1, 
                    'en_conv_cache2': en_conv_cache2, 'en_res_cache2': en_res_cache2, 
                    'en_conv_cache3': en_conv_cache3, 'en_res_cache3': en_res_cache3,
                    'en_conv_cache4': en_conv_cache4, 'en_res_cache4': en_res_cache4, 
                    'en_conv_cache5': en_conv_cache5, 'en_res_cache5': en_res_cache5,
                    'h_cache': h_cache, 
                    'de_conv_cache5': de_conv_cache5, 'de_res_cache5': de_res_cache5, 
                    'de_conv_cache4': de_conv_cache4, 'de_res_cache4': de_res_cache4, 
                    'de_conv_cache3': de_conv_cache3, 'de_res_cache3': de_res_cache3,
                    'de_conv_cache2': de_conv_cache2, 'de_res_cache2': de_res_cache2, 
                    'de_conv_cache1': de_conv_cache1, 'de_res_cache1': de_res_cache1,
                    'm_cache': m_cache})

        toc = time.perf_counter()
        T_list.append(toc-tic)
        outputs.append(out_i)

    print(">>> inference time: mean: {:.1f}ms, max: {:.1f}ms, min: {:.1f}ms".format(1e3*np.mean(T_list), 1e3*np.max(T_list), 1e3*np.min(T_list)))

    # 将所有输出帧拼接起来
    outputs = np.concatenate(outputs, axis=-2)  # 形状: (1, 257, T, 2)

    # 将输出转换为复数形式
    output_real = outputs[0, :, :, 0]  # 实部
    output_imag = outputs[0, :, :, 1]  # 虚部
    output_complex = output_real + 1j * output_imag

    # 计算ISTFT以重建时域信号
    t, enhanced_audio = signal.istft(output_complex, fs, nperseg=n_fft, noverlap=n_fft-hop_length, window='hann')

    # 确保长度与原始音频相同
    if len(enhanced_audio) > len(x):
        enhanced_audio = enhanced_audio[:len(x)]
    elif len(enhanced_audio) < len(x):
        enhanced_audio = np.pad(enhanced_audio, (0, len(x) - len(enhanced_audio)))

    # 保存增强后的音频
    sf.write('test_enhanced.wav', enhanced_audio, fs)
    print("增强后的音频已保存为 'test_enhanced.wav'")