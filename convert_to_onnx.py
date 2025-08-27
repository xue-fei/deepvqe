import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# 保持原有的模型定义不变
class FE(nn.Module):
    """Feature extraction"""
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c
    def forward(self, x):
        """x: (B,F,T,2)"""
        x_mag = torch.sqrt(x[...,[0]]**2 + x[...,[1]]**2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1-self.c) + 1e-12)
        return x_c.permute(0,3,2,1).contiguous()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4,3))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()
    def forward(self, x):
        """x: (B,C,T,F)"""
        y = self.elu(self.bn(self.conv(self.pad(x))))
        return y + x
        
class AlignBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, delay=100):
        super().__init__()
        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)
        self.unfold = nn.Sequential(nn.ZeroPad2d([0,0,delay-1,0]),
                                    nn.Unfold((delay, 1)))
        self.conv = nn.Sequential(nn.ZeroPad2d([1,1,4,0]),
                                  nn.Conv2d(hidden_channels, 1, (5,3)))
        
    def forward(self, x_mic, x_ref):
        """
        x_mic: (B,C,T,F)
        x_ref: (B,C,T,F)
        """
        Q = self.pconv_mic(x_mic)  # (B,H,T,F)
        K = self.pconv_ref(x_ref)  # (B,H,T,F)
        Ku = self.unfold(K)        # (B, H*D, T*F)
        Ku = Ku.view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3])\
            .permute(0,1,3,2,4).contiguous()  # (B,H,T,D,F)
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)      # (B,H,T,D)
        V = self.conv(V)           # (B,1,T,D)
        A = torch.softmax(V, dim=-1)[..., None]  # (B,1,T,D,1)
        
        y = self.unfold(x_ref).view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3])\
                .permute(0,1,3,2,4).contiguous()  # (B,H,T,D,F)
        y = torch.sum(y * A, dim=-2)
        return y

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), stride=(1,2)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.resblock = ResidualBlock(out_channels)
    def forward(self, x):
        return self.resblock(self.elu(self.bn(self.conv(self.pad(x)))))

class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        """x : (B,C,T,F)"""
        y = rearrange(x, 'b c t f -> b t (c f)')
        y = self.gru(y)[0]
        y = self.fc(y)
        y = rearrange(y, 'b t (c f) -> b c t f', c=x.shape[1])
        return y
    
class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels*2, kernel_size)
        
    def forward(self, x):
        y = self.conv(self.pad(x))
        y = rearrange(y, 'b (r c) t f -> b c t (r f)', r=2)
        return y
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), is_last=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.is_last = is_last
    def forward(self, x, x_en):
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y
    
class CCM(nn.Module):
    """Complex convolving mask block"""
    def __init__(self):
        super().__init__()
        self.v = torch.tensor([[1,        -1/2,           -1/2],
                               [0, np.sqrt(3)/2, -np.sqrt(3)/2]], dtype=torch.float32)  # (2,3)
        self.unfold = nn.Sequential(nn.ZeroPad2d([1,1,2,0]),
                                    nn.Unfold(kernel_size=(3,3)))
    
    def forward(self, m, x):
        """
        m: (B,27,T,F)
        x: (B,F,T,2)"""
        m = rearrange(m, 'b (r c) t f -> b r c t f', r=3)
        H_real = torch.sum(self.v[0].to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)
        H_imag = torch.sum(self.v[1].to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)

        M_real = rearrange(H_real, 'b (m n) t f -> b m n t f', m=3)  # (B,3,3,T,F)
        M_imag = rearrange(H_imag, 'b (m n) t f -> b m n t f', m=3)  # (B,3,3,T,F)
        
        x = x.permute(0,3,2,1).contiguous()  # (B,2,T,F)
        x_unfold = self.unfold(x)
        x_unfold = rearrange(x_unfold, 'b (c m n) (t f) -> b c m n t f', m=3,n=3,f=x.shape[-1])

        x_enh_real = torch.sum(M_real * x_unfold[:,0] - M_imag * x_unfold[:,1], dim=(1,2))  # (B,T,F)
        x_enh_imag = torch.sum(M_real * x_unfold[:,1] + M_imag * x_unfold[:,0], dim=(1,2))  # (B,T,F)
        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3).transpose(1,2).contiguous()
        return x_enh

class DeepVQE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = FE()
        self.enblock1 = EncoderBlock(2, 64)
        self.enblock2 = EncoderBlock(64, 128)
        self.enblock3 = EncoderBlock(128, 128)
        self.enblock4 = EncoderBlock(128, 128)
        self.enblock5 = EncoderBlock(128, 128)
        
        self.bottle = Bottleneck(128*9, 64*9)
        
        self.deblock5 = DecoderBlock(128, 128)
        self.deblock4 = DecoderBlock(128, 128)
        self.deblock3 = DecoderBlock(128, 128)
        self.deblock2 = DecoderBlock(128, 64)
        self.deblock1 = DecoderBlock(64, 27)
        self.ccm = CCM()
        
    def forward(self, x):
        """x: (B,F,T,2)"""
        en_x0 = self.fe(x)            # ; print(en_x0.shape)
        en_x1 = self.enblock1(en_x0)  # ; print(en_x1.shape)
        en_x2 = self.enblock2(en_x1)  # ; print(en_x2.shape)
        en_x3 = self.enblock3(en_x2)  # ; print(en_x3.shape)
        en_x4 = self.enblock4(en_x3)  # ; print(en_x4.shape)
        en_x5 = self.enblock5(en_x4)  # ; print(en_x5.shape)

        en_xr = self.bottle(en_x5)    # ; print(en_xr.shape)
        
        de_x5 = self.deblock5(en_xr, en_x5)[..., :en_x4.shape[-1]]  # ; print(de_x5.shape)
        de_x4 = self.deblock4(de_x5, en_x4)[..., :en_x3.shape[-1]]  # ; print(de_x4.shape)
        de_x3 = self.deblock3(de_x4, en_x3)[..., :en_x2.shape[-1]]  # ; print(de_x3.shape)
        de_x2 = self.deblock2(de_x3, en_x2)[..., :en_x1.shape[-1]]  # ; print(de_x2.shape)
        de_x1 = self.deblock1(de_x2, en_x1)[..., :en_x0.shape[-1]]  # ; print(de_x1.shape)
        
        x_enh = self.ccm(de_x1, x)  # (B,F,T,2)
        
        return x_enh

# 加载模型权重函数
def load_model(checkpoint_path):
    """加载预训练模型权重"""
    model = DeepVQE()
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 如果检查点包含模型状态字典
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 假设整个检查点就是模型状态字典
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model

# 导出为ONNX格式
def export_to_onnx(model, onnx_path, input_shape=(1, 257, 63, 2)):
    """将模型导出为ONNX格式"""
    # 创建示例输入
    dummy_input = torch.randn(*input_shape)
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,  # 使用较新的opset版本以支持更多操作
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {
                0: 'batch_size',
                2: 'sequence_length'  # 时间维度可以是动态的
            },
            'output': {
                0: 'batch_size',
                2: 'sequence_length'
            }
        }
    )
    print(f"模型已成功导出到 {onnx_path}")

# 验证ONNX模型
def validate_onnx(onnx_path, input_shape=(1, 257, 63, 2)):
    """验证导出的ONNX模型"""
    import onnx
    import onnxruntime as ort
    
    # 加载ONNX模型
    onnx_model = onnx.load(onnx_path)
    
    # 验证模型
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证成功!")
    
    # 使用ONNX Runtime测试推理
    ort_session = ort.InferenceSession(onnx_path)
    
    # 准备输入
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 运行推理
    outputs = ort_session.run(None, {'input': dummy_input})
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {outputs[0].shape}")
    print("ONNX模型推理测试成功!")

if __name__ == "__main__":
    # 参数设置
    checkpoint_path = "deepvqe_trained_on_DNS3.tar"  # 您的模型权重文件路径
    onnx_path = "deepvqe.onnx"  # 输出的ONNX文件路径
    
    # 步骤1: 加载模型
    print("步骤1: 加载模型...")
    model = load_model(checkpoint_path)
    print("模型加载成功!")
    
    # 步骤2: 导出为ONNX格式
    print("步骤2: 导出为ONNX格式...")
    export_to_onnx(model, onnx_path)
    
    # 步骤3: 验证ONNX模型
    print("步骤3: 验证ONNX模型...")
    validate_onnx(onnx_path)
    
    print("全部步骤完成! DeepVQE模型已成功转换为ONNX格式。")