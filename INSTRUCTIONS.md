# TMUX

# Cuda

### Killing processes on the GPU

Replace `$PID` with the PID shown in `nivida-smi` and then type

`nvidia-smi | grep 'python' | awk '{ print $PID }' | xargs -n1 kill -9`
