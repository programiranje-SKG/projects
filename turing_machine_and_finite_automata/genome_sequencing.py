# Koncni avtomat, ki bo v genomu poiskal vse pojavitve vzorcev AAUAUG in AACAUG.

delta = [
    dict(A=1, C=0, G=0, U=0),  # 0
    dict(A=2, C=0, G=0, U=0),  # 1
    dict(A=2, C=3, G=0, U=3),  # 2
    dict(A=4, C=0, G=0, U=0),  # 3
    dict(A=2, C=0, G=0, U=5),  # 4
    dict(A=1, C=0, G=6, U=0),  # 5
    dict(A=1, C=0, G=0, U=0),  # 6
]


def read_genome(path):
    with open(path, 'r') as file:
        return file.readline().strip()


def next_state(curr_state: int, input_letter: str):
    return delta[curr_state][input_letter]


def find_locations(genome):
    # find AAUAUG or AACAUG
    s_len = 6
    # location is index of start of substring in genome
    locs = []
    state = 0
    for i, letter in enumerate(genome):
        state = next_state(state, letter)
        if state == 6:
            # save index
            locs.append(i - s_len + 1)
    print(f"Total hits: {len(locs)}")
    for loc in locs:
        print(f"{loc}: {genome[loc: loc + s_len]}")
    return locs


if __name__ == '__main__':
    genome = read_genome("generated_genome.txt")
    locations = find_locations(genome)
    print(locations)
