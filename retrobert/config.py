import argparse


# Default arguments string
ARGS_STR = f"""
--exp_name=retrobert \
--train_epochs=20 \
--train_batch_size=64 \
--valid_batch_size=64 \
--test_batch_size=64 \
--gradient_accumulation_steps=2 \
--learning_rate=8e-7 \
--warmup_steps=100 \
--weight_decay=0.1 \
--adam_epsilon=1e-8 \
--seed=42 \
--max_seq_length=64 \
--max_grad_norm=1.0 \
"""


def add_default_args(parser):
    """
    Define and set default arguments for the script.
    """
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--save_every_epoch', type=bool, default=False)
    parser.add_argument('--report_every_step', type=int, default=10)
    parser.add_argument('--eval_every_step', type=int, default=10)
    parser.add_argument('--max_seq_length', type=int, default=64)
    return parser

