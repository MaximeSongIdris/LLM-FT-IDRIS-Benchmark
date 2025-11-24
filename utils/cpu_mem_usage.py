
## mememory usage
def get_proc_status(keys = None):
    """Get value from keys from /proc/self/status file"""
    with open('/proc/self/status') as f:
        data = dict(map(str.strip, line.split(':', 1)) for line in f)
    return tuple(data[k] for k in keys) if keys else data

def memory_usage():
    """Memory usage of the current process."""
    try:
        peak, hwm = get_proc_status(('VmPeak', 'VmHWM'))
        print("VmPeak: ",peak," VmHWM: ", hwm)
    except:
        print("Get Memory Usage: Get value from keys from /proc/self/status file is uncallable")