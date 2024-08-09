mkdir -p hw_run/traces/vulkan
curl https://zenodo.org/records/13287587/files/spl_vio.tar.gz?download=1 --output spl_vio.tar.gz
tar -xvf spl_vio.tar.gz
mv spl_vio/* hw_run/traces/vulkan
rm -rf spl_vio
