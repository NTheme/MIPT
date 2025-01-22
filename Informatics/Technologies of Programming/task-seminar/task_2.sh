START=`date +%s.%N`
$1
END=`date +%s.%N`
echo $(echo ${END} - ${START} | bc)
