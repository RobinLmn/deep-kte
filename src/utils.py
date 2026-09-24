import torch


def to_tensor(*arrays, device=None):

    def convert(outcomes):
        outcome_tensor = torch.tensor(outcomes, dtype=torch.float32, device=device)
        if outcome_tensor.ndim in (1, 3):
            outcome_tensor = outcome_tensor.unsqueeze(1)
        elif outcome_tensor.ndim == 4:
            outcome_tensor = outcome_tensor.permute(0, 3, 1, 2)
        return outcome_tensor

    tensors = tuple(convert(array) for array in arrays)
    return tensors[0] if len(tensors) == 1 else tensors


def pdist2(first_outcomes, second_outcomes, distance_float64=False):
    output_dtype = first_outcomes.dtype

    if distance_float64:
        (first_outcomes, second_outcomes) = (first_outcomes.double(), second_outcomes.double())

    first_norm = (first_outcomes**2).sum(1).view(-1, 1)
    second_norm = (second_outcomes**2).sum(1).view(1, -1)
    squared_distances = first_norm + second_norm - 2.0 * torch.mm(first_outcomes, torch.transpose(second_outcomes, 0, 1))
    squared_distances[squared_distances < 0] = 0
    return squared_distances.to(output_dtype) if distance_float64 else squared_distances
