import winsound

def beeps(n=3):
    '''Produce n beeps. Default: 3'''
    for _ in range(n):
        winsound.Beep(200,500)