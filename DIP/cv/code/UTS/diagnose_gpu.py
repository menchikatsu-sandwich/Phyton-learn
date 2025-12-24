import sys
import subprocess

def safe_import(name):
    try:
        m = __import__(name)
        return m, None
    except Exception as e:
        return None, str(e)

print('Python:', sys.version.replace('\n',' '))

m, err = safe_import('torch')
if m:
    try:
        cuda_ok = m.cuda.is_available()
        dev_count = m.cuda.device_count()
    except Exception as e:
        cuda_ok = f'error checking: {e}'
        dev_count = 'n/a'
    print('PyTorch:', m.__version__, 'cuda_available=', cuda_ok, 'device_count=', dev_count)
else:
    print('PyTorch import error:', err)

m, err = safe_import('tensorflow')
if m:
    try:
        gpus = m.config.list_physical_devices('GPU')
        print('TensorFlow:', m.__version__, 'GPUs found=', len(gpus), gpus)
    except Exception as e:
        print('TensorFlow import error checking GPUs:', e)
else:
    print('TensorFlow import error:', err)

m, err = safe_import('cv2')
if m:
    try:
        cuda_attr = hasattr(m, 'cuda')
        cuda_count = m.cuda.getCudaEnabledDeviceCount() if cuda_attr else 'no_cuda_attr'
        print('OpenCV:', m.__version__, 'cuda_attr=', cuda_attr, 'cuda_enabled_count=', cuda_count)
    except Exception as e:
        print('OpenCV import error checking cuda:', e)
else:
    print('OpenCV import error:', err)

# nvidia-smi
try:
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu,utilization.memory,memory.total,memory.used', '--format=csv,noheader,nounits'], universal_newlines=True)
    print('\nnvidia-smi output:')
    print(out)
except Exception as e:
    print('\nnvidia-smi not available or error:', e)
