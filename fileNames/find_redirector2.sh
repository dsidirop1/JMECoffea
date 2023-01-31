# set the cvmfs environment
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh
export SSL_CERT_DIR='/etc/pki/tls/certs:/etc/grid-security/certificates'
in_file=$1
out_file=$2
LINES=$(cat $in_file)

for FILE in $LINES
do
	site_aliases=$(dasgoclient -query="site file=$FILE")
	correct_alias=

	if [[ $(echo ${site_aliases} | head -n 1 )  = WARNING* ]] ; then
     echo "Discarded file = " $FILE
	else
		for site_alias in ${site_aliases}
		do 
			if [[ ${site_alias} == *_Pisa ]]; then
				correct_alias=${site_alias}
				break
			fi
		done
	
   	if [[ -z "$correct_alias" ]]; then
    		for site_alias in ${site_aliases}
    		do 
    			echo Checking alias ${site_alias}
    			if [[ ${site_alias} != *_Disk ]]; then
    				echo alias is good
    				correct_alias=${site_alias}
    				break
    			fi
          done
		fi

		if [[ -z "$correct_alias" ]]; then
			echo "Only on disk. Discarding file = " $FILE
			break
		fi
		
		if [[ ${correct_alias} == *_Pisa ]]; then
			redirector=root://stormgf2.pi.infn.it:1094/
		else
			redirector=$(python GetSiteInfo.py ${correct_alias} | grep XROOTD | tr -s ' ' | awk '{ print $NF }')
		fi
		full_name=${redirector}//${FILE}
		echo file = ${FILE}, site = ${correct_alias}, redirector = ${redirector}
		#echo full name =  $full_name
		echo $full_name >> $out_file
	fi
done

