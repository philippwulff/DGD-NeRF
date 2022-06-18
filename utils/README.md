# TMUX

# Cuda

Check GPU usage: `nvidia-smi`

Find user who is running a process with some PID, e.g. 1234: `ps -u -p 1234`
### Killing processes on the GPU

Replace `$PID` with the PID shown in `nivida-smi` and then type

`nvidia-smi | grep 'python' | awk '{ print $PID }' | xargs -n1 kill -9`
