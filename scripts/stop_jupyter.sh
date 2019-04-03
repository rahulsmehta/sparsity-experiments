PID=`cat jupyter.pid`
kill -9 $PID
rm jupyter.pid
