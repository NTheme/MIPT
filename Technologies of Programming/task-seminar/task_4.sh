for i in "$@"
do
	if [[ ! -d $i ]]
	then
		echo $(file $i)
	fi
done
