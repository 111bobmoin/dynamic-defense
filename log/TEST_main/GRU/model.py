import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUWithAttention(nn.Module):
    def __init__(self, input_size=29, hidden_size=32, num_layers=1, output_size=1):
        super(GRUWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU层（保持原始结构）
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False
        )

        # 注意力机制层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

        # 自动获取设备信息
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # 确保模型参数在正确设备上

    def forward(self, x, hidden):
        # 确保输入数据与模型在同一设备
        x = x.to(self.device)

        # GRU输出
        gru_out, hidden = self.gru(x, hidden)

        # 注意力计算
        attn_weights = F.softmax(
            self.attention(gru_out).squeeze(2),  # (seq_len, batch)
            dim=0
        )

        # 上下文向量计算
        context = torch.sum(attn_weights.unsqueeze(2) * gru_out, dim=0)

        # 最终输出
        out = self.fc(context)
        return torch.sigmoid(out.squeeze()), hidden

    def init_hidden(self, batch_size):
        # 使用模型自身的设备信息
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)


# 测试用例（修复设备问题）
if __name__ == "__main__":
    # 初始化模型并自动置于正确设备
    model = GRUWithAttention()

    # 测试数据生成在模型相同的设备
    test_input = torch.randn(5, 3, 29).to(model.device)  # seq_len=5, batch=3
    hidden = model.init_hidden(batch_size=3)

    outputs, _ = model(test_input, hidden)
    print(f"Device: {model.device}")
    print(f"Output shape: {outputs.shape}")
    print(f"Sample predictions:\n{outputs}")