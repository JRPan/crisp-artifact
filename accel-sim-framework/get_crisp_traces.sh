mkdir -p hw_run/traces/vulkan
wget http://tgrogers-pc02.ecn.purdue.edu/crisp/spl_vio.tar.gz
tar -xvf spl_vio.tar.gz
mv spl_vio/* hw_run/traces/vulkan
rm -rf spl_vio