# set the cvmfs environment
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh
export SSL_CERT_DIR='/etc/pki/tls/certs:/etc/grid-security/certificates'
in_file=$1
out_file=$2
LINES=$(cat $in_file)

for FILE in $LINES
do
   site_alias=$(dasgoclient -query="site file=$FILE" | head -n 1)
	if [[ $outp = WARNING* ]] ; then
     echo "Discarded file = " $FILE
	else
		redirector=$(python GetSiteInfo.py ${site_alias} | grep XROOTD | tr -s ' ' | cut -d ' ' -f 3)
		# redirector=${redirector%?}
		full_name=${redirector}//${FILE}
		#echo file = ${FILE}, site = ${site_alias}, redirector = ${redirector}
		#echo full name =  $full_name
		echo $full_name >> $out_file
	fi
done

