import time
import numpy as np
import soundfile as sf
import onnxruntime as ort
from scipy.signal import stft, istft

def main():
    # 加载ONNX模型
    model_path = 'onnx_models/deepvqe_simple.onnx'  # 替换为您的ONNX模型路径
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 读取音频文件
    audio_path = '../test.wav'  # 替换为您的音频文件路径
    x, fs = sf.read(audio_path, dtype='float32')  # x: (n_samples,), 单声道, 16 kHz
    # 确保音频是单声道
    if len(x.shape) > 1:
        x = np.mean(x, axis=1)
    
    # 参数设置
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = np.hanning(win_length)
    
    # 计算总帧数
    n_frames = 1 + (len(x) - n_fft) // hop_length
    print(f"音频长度: {len(x)} 样本, 帧数: {n_frames}")
    
    # 初始化输出数组
    enhanced_audio = np.zeros(len(x))
    
    # 初始化缓存 - 关键修正：确保缓存形状与模型输入要求一致
    en_conv_cache1 = np.zeros([1, 2, 3, 257], dtype=np.float32)
    en_res_cache1  = np.zeros([1, 64, 3, 129], dtype=np.float32)
    en_conv_cache2 = np.zeros([1, 64, 3, 129], dtype=np.float32)
    en_res_cache2  = np.zeros([1, 128, 3, 65], dtype=np.float32)
    en_conv_cache3 = np.zeros([1, 128, 3, 65], dtype=np.float32)
    en_res_cache3  = np.zeros([1, 128, 3, 33], dtype=np.float32)
    en_conv_cache4 = np.zeros([1, 128, 3, 33], dtype=np.float32)
    en_res_cache4  = np.zeros([1, 128, 3, 17], dtype=np.float32)
    en_conv_cache5 = np.zeros([1, 128, 3, 17], dtype=np.float32)
    en_res_cache5  = np.zeros([1, 128, 3, 9], dtype=np.float32)
    h_cache        = np.zeros([1, 1, 64*9], dtype=np.float32)
    de_res_cache5  = np.zeros([1, 128, 3, 9], dtype=np.float32)
    de_conv_cache5 = np.zeros([1, 128, 3, 9], dtype=np.float32)
    de_res_cache4  = np.zeros([1, 128, 3, 17], dtype=np.float32)
    de_conv_cache4 = np.zeros([1, 128, 3, 17], dtype=np.float32)
    de_res_cache3  = np.zeros([1, 128, 3, 33], dtype=np.float32)
    de_conv_cache3 = np.zeros([1, 128, 3, 33], dtype=np.float32)
    de_res_cache2  = np.zeros([1, 128, 3, 65], dtype=np.float32)
    de_conv_cache2 = np.zeros([1, 128, 3, 65], dtype=np.float32)
    de_res_cache1  = np.zeros([1, 64, 3, 129], dtype=np.float32)
    de_conv_cache1 = np.zeros([1, 64, 3, 129], dtype=np.float32)
    m_cache        = np.zeros([1, 257, 2, 2], dtype=np.float32)
    
    # 获取并打印输入信息（关键：检查输入形状是否匹配）
    input_info = session.get_inputs()
    print("模型输入信息:")
    for info in input_info:
        print(f"  {info.name}: 形状{info.shape}, 类型{info.type}")
    
    # 获取输出名称
    output_names = [output.name for output in session.get_outputs()]
    
    # 逐帧处理
    start_time = time.time()
    
    for i in range(n_frames):
        # 提取当前帧
        start = i * hop_length
        end = start + n_fft
        
        # 如果帧超出音频范围，填充零
        if end > len(x):
            frame = np.zeros(n_fft, dtype=np.float32)
            frame[:len(x)-start] = x[start:]
        else:
            frame = x[start:end].astype(np.float32)
        
        # 应用窗函数
        frame_windowed = frame * window
        
        # 计算当前帧的STFT
        _, _, Zxx = stft(frame_windowed, fs=fs, nperseg=n_fft, noverlap=0, 
                        nfft=n_fft, window='hann', boundary=None)
        
        # 获取频率bin（只取前257个）
        stft_frame = Zxx[:257, 0]  # 形状: (257,)
        
        # 分离实部和虚部并转换为正确形状
        stft_real = np.real(stft_frame)
        stft_imag = np.imag(stft_frame)
        
        # 组合为模型输入格式 (1, 257, 1, 2) - 确保与模型输入形状匹配
        stft_input = np.zeros((1, 257, 1, 2), dtype=np.float32)
        stft_input[0, :, 0, 0] = stft_real
        stft_input[0, :, 0, 1] = stft_imag
        
        # 运行模型推理 - 关键修正：确保输入字典正确映射
        inputs = {
            'mix': stft_input,
            'en_conv_cache1': en_conv_cache1, 
            'en_res_cache1': en_res_cache1, 
            'en_conv_cache2': en_conv_cache2, 
            'en_res_cache2': en_res_cache2, 
            'en_conv_cache3': en_conv_cache3, 
            'en_res_cache3': en_res_cache3,
            'en_conv_cache4': en_conv_cache4, 
            'en_res_cache4': en_res_cache4, 
            'en_conv_cache5': en_conv_cache5, 
            'en_res_cache5': en_res_cache5,
            'h_cache': h_cache, 
            'de_conv_cache5': de_conv_cache5, 
            'de_res_cache5': de_res_cache5, 
            'de_conv_cache4': de_conv_cache4, 
            'de_res_cache4': de_res_cache4, 
            'de_conv_cache3': de_conv_cache3, 
            'de_res_cache3': de_res_cache3,
            'de_conv_cache2': de_conv_cache2, 
            'de_res_cache2': de_res_cache2, 
            'de_conv_cache1': de_conv_cache1, 
            'de_res_cache1': de_res_cache1,
            'm_cache': m_cache
        }
        
        # 确保只传入模型需要的输入
        filtered_inputs = {k: v for k, v in inputs.items() if k in [info.name for info in input_info]}
        
        try:
            outputs = session.run(output_names, filtered_inputs)
        except Exception as e:
            print(f"处理第{i}帧时出错: {str(e)}")
            print(f"输入'mix'形状: {stft_input.shape}")
            raise
        
        # 提取输出并更新缓存
        y_stft = outputs[0]  # 增强后的STFT
        
        # 更新缓存（根据实际输出顺序调整索引）
        en_conv_cache1 = outputs[1]
        en_res_cache1 = outputs[2]
        en_conv_cache2 = outputs[3]
        en_res_cache2 = outputs[4]
        en_conv_cache3 = outputs[5]
        en_res_cache3 = outputs[6]
        en_conv_cache4 = outputs[7]
        en_res_cache4 = outputs[8]
        en_conv_cache5 = outputs[9]
        en_res_cache5 = outputs[10]
        h_cache = outputs[11]
        de_conv_cache5 = outputs[12]
        de_res_cache5 = outputs[13]
        de_conv_cache4 = outputs[14]
        de_res_cache4 = outputs[15]
        de_conv_cache3 = outputs[16]
        de_res_cache3 = outputs[17]
        de_conv_cache2 = outputs[18]
        de_res_cache2 = outputs[19]
        de_conv_cache1 = outputs[20]
        de_res_cache1 = outputs[21]
        m_cache = outputs[22]
        
        # 将输出转换为复数形式
        y_real = y_stft[0, :, 0, 0]
        y_imag = y_stft[0, :, 0, 1]
        y_complex = y_real + 1j * y_imag
        
        # 重建完整的频谱
        full_spectrum = np.zeros(n_fft, dtype=complex)
        full_spectrum[:257] = y_complex
        full_spectrum[257:] = np.conj(y_complex[1:256][::-1])  # 对称填充
        
        # 计算逆变换
        _, y_frame = istft(
            full_spectrum.reshape(-1, 1), 
            fs=fs, 
            nperseg=n_fft, 
            noverlap=0, 
            nfft=n_fft, 
            window='hann',
            boundary=None
        )
        
        # 应用窗函数并添加到输出（重叠相加）
        y_frame_windowed = y_frame * window
        
        # 确保长度正确
        if len(y_frame_windowed) > n_fft:
            y_frame_windowed = y_frame_windowed[:n_fft]
        
        # 添加到输出（重叠相加）
        enhanced_audio[start:start+len(y_frame_windowed)] += y_frame_windowed
        
        # 打印进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{n_frames} 帧")
    
    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time
    real_time_factor = processing_time / (len(x) / fs)
    print(f"处理完成，耗时: {processing_time:.2f}秒")
    print(f"实时因子: {real_time_factor:.2f} (小于1表示快于实时)")
    
    # 归一化输出
    max_val = np.max(np.abs(enhanced_audio))
    if max_val > 0:
        enhanced_audio = enhanced_audio / max_val
    
    # 保存增强后的音频
    output_path = 'test_enhanced.wav'
    sf.write(output_path, enhanced_audio, fs)
    print(f"增强后的音频已保存为: {output_path}")

if __name__ == "__main__":
    main()
