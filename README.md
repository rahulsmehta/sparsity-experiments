ResNet Experiments
=============================

# Connecting to AWS
Ensure that the following environment variables are set (easiest way to do this is consolidate these in an `~/.aws_config` file that is read in your `bash_profile`). For instance:

```
export AWS_KEY_PATH=/path/to/my_key.pem
export AWS_HOST=<my_host>.compute-<n>.amazonaws.com
export AWS_USER=ec2-user
```

Once this is complete run `./connect-ssh.sh [-o]`. The optional flag `-o` forwards `AWS_HOST:8888` to `localhost:8888` to use Jupyter Lab.
