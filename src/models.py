import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        filters=64,
        kernel_sizes=(3, 4, 5),
        num_classes=3,
        dropout=0.4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=filters,
                kernel_size=kernel,
            )
            for kernel in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            filters * len(kernel_sizes),
            num_classes,
        )

    def forward(self, tokens, lengths):
        # [batch, tokens, embedding] -> [batch, embedding, tokens]
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled_features = []

        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            kernel = convolution.kernel_size[0]

            # Ignore windows extending beyond a sequence's real tokens.
            # For very short sequences, allow one padded window.
            valid_windows = (lengths - kernel + 1).clamp(min=1)
            positions = torch.arange(
                features.size(2),
                device=features.device,
            )

            valid = (
                positions.unsqueeze(0)
                < valid_windows.to(features.device).unsqueeze(1)
            )

            features = features.masked_fill(
                ~valid.unsqueeze(1),
                float("-inf"),
            )

            pooled_features.append(features.max(dim=2).values)

        combined = torch.cat(pooled_features, dim=1)
        return self.classifier(self.dropout(combined))


class TextLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_size=64,
        num_classes=3,
        dropout=0.4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)

        # Packing prevents padding from changing the recurrent states.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        # Final forward and backward states of the single LSTM layer.
        combined = torch.cat(
            [hidden[-2], hidden[-1]],
            dim=1,
        )

        return self.classifier(self.dropout(combined))


def check_models():
    """Check output shapes, gradients, and padding behavior."""
    torch.manual_seed(42)

    tokens = torch.tensor([
        [4, 5, 6, 7, 8, 9, 0],
        [4, 5, 0, 0, 0, 0, 0],
    ])
    lengths = torch.tensor([6, 2])
    labels = torch.tensor([0, 2])

    for model_class in [TextCNN, TextLSTM]:
        model = model_class(vocab_size=20)
        model.eval()

        with torch.no_grad():
            logits = model(tokens, lengths)

            extra_padding = torch.zeros(
                (tokens.size(0), 4),
                dtype=torch.long,
            )
            padded_logits = model(
                torch.cat([tokens, extra_padding], dim=1),
                lengths,
            )

        assert logits.shape == (2, 3)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(
            logits,
            padded_logits,
            atol=1e-6,
            rtol=1e-5,
        )

        model.train()
        loss = nn.CrossEntropyLoss()(model(tokens, lengths), labels)
        loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()

        print(
            f"{model_class.__name__}: "
            "shape, gradient, and padding checks passed."
        )


if __name__ == "__main__":
    check_models()