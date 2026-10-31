import torch


def pytest_sessionstart(session) -> None:
    # Keep tiny CPU tests responsive and predictable on shared runners.
    torch.set_num_threads(1)
