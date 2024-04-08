def select_mode(key, mode):
    if key == ord('r'):
        return 1, False
    elif key == ord('c'):
        return mode, True
    else:
        return mode, False
