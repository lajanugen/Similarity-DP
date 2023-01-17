from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np

def compute_alignment_from_trajectories(trajectory1, trajectory2, cross_encoder=True):

    # trajectory1 is a list of phrases/sentences
    # trajectory2 is a list of phrases/sentences

    if cross_encoder:
        model = CrossEncoder('cross-encoder/stsb-roberta-large', device='cuda')
    else:
        model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

    scores = compute_similarity(trajectory1, trajectory2, model, cross_encoder=cross_encoder)
    alignment = compute_alignment_from_similarity(scores, allow_repetition=False)

    return ailgnment

def compute_similarity(trajectory1, trajectory2, model, cross_encoder=True):

    if cross_encoder:
        scores = np.array([model.predict([(x, y) for y in trajetory2]) for x in trajectory1])

    else:
        sent_embs = model.encode(trajectory1, show_progress_bar=False)
        subtask_embs = model.encode(trajectory2, show_progress_bar=False)
        scores = util.cos_sim(sent_embs, subtask_embs)

    return scores

def compute_alignment_from_similarity(sim, allow_repetition=False):

    # Given similarity matrix sim, compute optimal alignment

    dp_array = np.ones(sim.shape) * -1
    dp_backtrack = np.zeros(sim.shape).astype(np.int)
    dp_array[0, 0] = sim[0, 0]
    for j in range(sim.shape[1]):
        for i in range(1, sim.shape[0]):
            if allow_repetition:
                prev = dp_array[i][j-1] if j > 0 else 0
            else:
                prev = dp_array[i-1][j-1] if j > 0 else 0
            dp_array[i][j] = max(sim[i][j] + prev, dp_array[i-1][j])
            if sim[i][j] + prev > dp_array[i-1][j]:
                dp_backtrack[i][j] = i
            else:
                dp_backtrack[i][j] = dp_backtrack[i-1][j]

    assignment = []
    i = len(sim) - 1
    for j in range(sim.shape[1]-1, -1, -1):
        assignment.append(dp_backtrack[i][j])
        if allow_repetition:
            i = dp_backtrack[i][j]
        else:
            i = dp_backtrack[i][j] - 1
    assignment = assignment[::-1]

    return assignment
