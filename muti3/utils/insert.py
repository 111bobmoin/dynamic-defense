import time


def attack(model: str, attack: str):
    print(f"对原异构体 {model.upper()} 进行 {attack} 攻击")
    time.sleep(3)
    print(f"攻击结束, 攻击后异构体存储在/model/attack_{model}_{attack}.pth中")

