MAXLENGTH=10000
CPATH="$1//task1"

rm -rf ${CPATH}
mkdir ${CPATH}

MAXSIZE=0
MAXINDEX=0
for ((i=1; i<=$2; i++))
do
	let "NUM = ${RANDOM} % ${MAXLENGTH}"
	cat /dev/urandom | tr -dc A-Za-z0-9 | head -c$NUM | tee >> "${CPATH}//${i}.txt"
	
	NEWSIZE=`stat -c %s ${CPATH}//${i}.txt`
	if [[ $MAXSIZE < $NEWSIZE ]]
	then
		MAXSIZE=${NEWSIZE}
		MAXINDEX=${i}
	fi
done

OUTFILE=$(date -r ${CPATH}//${MAXINDEX}.txt)_${MAXINDEX}_${MAXSIZE}.txt

for ((i=1; i<=$2; i++))
do
	cat ${CPATH}//${i}.txt >> "${OUTFILE}"
done	
