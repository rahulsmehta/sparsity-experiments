echo `which jupyter`
nohup jupyter lab > out.log 2>&1 &
echo $! > jupyter.pid

