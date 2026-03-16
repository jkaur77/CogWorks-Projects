from collections import Counter

def match(database, rec_fingerprints, rec_peak_times, matches_counter: Counter, songs) :
    for (fi, fj, dt), tm in zip(rec_fingerprints, rec_peak_times) :
        if ((fi, fj, dt) in database) :
            matching_tuples = database[(fi, fj, dt)]
            for time in matching_tuples:
                offset = tm - time[1]
                matches_counter[(time[0], offset)] += 1

    most_common = matches_counter.most_common(1)
    print(most_common)
    id = most_common[0][0][0]
    print(list(songs)[id])
