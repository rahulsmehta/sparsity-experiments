tunnel_flag='false'

print_usage() {
  echo "Usage: ./connect-ssh.sh [-o]"
}

while getopts 'o' flag; do
  case "${flag}" in
    o) tunnel_flag='true' ;;
    *) print_usage
       exit 1 ;;
  esac
done

if "$tunnel_flag" == "true" ; then
    echo "Opening local tunnel..."
    ssh -i $AWS_KEY_PATH -L 8888:localhost:8888 $AWS_USER@$AWS_HOST
else
    ssh -i $AWS_KEY_PATH $AWS_USER@$AWS_HOST

fi


