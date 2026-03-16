def add_to_database(fingerprints, peak_times, song_name, database, song_id_to_name):  # this would be given by method ,

    # return database
    # dictionary: 0: "willow", 1 : ""
    # generate new song id for this song -

    songId = len(song_id_to_name)
    song_id_to_name[songId] = song_name

    for (fm, fn, dt), tm in zip(fingerprints, peak_times):
        if (fm, fn, dt) not in database:
            database[(fm, fn, dt)] = []
        database[(fm, fn, dt)].append((songId, tm))

    return database, song_id_to_name
    # song id,
    # two arrays
    # first array- contains tuple - (fm, fn, dt)
    # second array- contains the time stamps of the peak that is associated with the fingerprint
    # fanoutn_m
