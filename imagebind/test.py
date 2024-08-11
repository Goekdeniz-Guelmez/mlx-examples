from models.transformer import SimpleTransformer, MultiheadAttention
from functools import partial

model = SimpleTransformer(
                embed_dim=1,
                num_blocks=2,
                ffn_dropout_rate=0.0,
                drop_path_rate=0.0,
                attn_target=partial(
                    MultiheadAttention,
                    1,
                    2,
                    bias=True
                )
            )

print(model)