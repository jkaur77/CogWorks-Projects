def db(fingerprints, peak_times, song_name, database, songs): # this would be given by method , 
    
    songId = songs[song_name][0]
    print(songId)
    
    for (fi, fj, dt), tm in zip(fingerprints, peak_times):
        
        if((fi, fj, dt) in database) :
            database[(fi, fj, dt)].append((songId, tm)) 
        else :
            database[(fi, fj, dt)] = [(songId, tm)]
    
    return database



        