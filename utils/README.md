# Useful Commands on the Server

### Git

Remove file only from remote repo: `git rm --cached file.txt`
Remove folder only from remote: `git rm -r --cached folder`
Then write the file/folder name in the `.gitignore` and push the changes.
### Tmux

Find all commands in the [Cheat Sheet](https://tmuxcheatsheet.com).

### Cuda

Check GPU usage: `nvidia-smi`

Get user who is running a process by its PID, e.g. 1234: `ps -u -p 1234`

Kill a process on the GPU given some PID, e.g. 1234: 
`nvidia-smi | grep 'python' | awk '{ print 1234 }' | xargs -n1 kill -9`

Change to another GPU, e.g. on index 15: `export CUDA_VISIBLE_DEVICES=15`


### NeRF Use-cases

- Image stabilization
