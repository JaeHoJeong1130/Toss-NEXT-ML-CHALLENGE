import torch

try:
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
        a = torch.tensor([1.0, 2.0]).cuda()
        print(f"GPU에서 간단한 연산 성공: {a}")
    else:
        print("CUDA를 사용할 수 없습니다.")
except Exception as e:
    print(f"오류 발생: {e}")