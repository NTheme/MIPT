PROJ_PATH=""
ARCH_NAME=""
CUR_PATH=$(realpath .)
declare -A ARG_COMPILER

#Parsing compiler argument and making an associative array exteision - compiler command
PARSE_COMPILER() {
  ARG=$1
  CMD=""
  STR=""
  for ((j=$((${#ARG}-1)); j>=0; j--))
  do
    if [ "${ARG: j:1}" == "=" ]
    then
      if [ "$CMD" == "" ]
      then
        CMD="${STR}"
      else
        ARG_COMPILER["${STR}"]="${CMD}"
      fi
      STR=""
    else
      STR="${ARG: j:1}${STR}"
    fi
  done
  ARG_COMPILER["${STR}"]="${CMD}"
}

#Parsing command line arguments
for ((i=1; i<=$#; i++))
do
  if [ "${@: i:1}" == "-s" ] || [ "${@: i:1}" == "--source" ]
  then
    i=$((i+1))	
    PROJ_PATH=$(realpath ${@: i:1})
  elif [ "${@: i:1}" == "-a" ] || [ "${@: i:1}" == "--archive" ]
  then
    i=$((i+1))
    ARCH_NAME="${@: i:1}"
  elif [ "${@: i:1}" == "-c" ] || [ "${@: i:1}" == "--compiler" ]
  then
    i=$((i+1))
    ARG="${@: i:1}"
    COMMAND=""
    PARSE_COMPILER "$ARG"
  fi
done

#Getting a firectory tree
cd ${PROJ_PATH}
PATH_TREE=($(find ${PROJ_PATH}))
declare -a TEMP_FILE

#Compiling appropriate files
for file in ${PATH_TREE[@]}
do
  RLTV_PATH=${file#$PROJ_PATH}
  if [ ! -d $file ] && [ -n "${file##*.}" ] && [ -n "${ARG_COMPILER[${file##*.}]}" ]
  then
    #Compiling...
    NEW_NAME=".$(dirname "${RLTV_PATH}")/$(basename "${file}" ${file##*.})exe"
    ${ARG_COMPILER[${file##*.}]} -o ${NEW_NAME} ${file}
    chmod +x ${NEW_NAME}

    #Creating a tree of temporary files to clean after
    if [ -e "${NEW_NAME}" ]
    then
      TEMP_FILE+=(${NEW_NAME})
    fi
  fi
done

#Archiving .exe
mkdir ${ARCH_NAME}
cp --parents ${TEMP_FILE[@]} ${ARCH_NAME}
tar cpf ${ARCH_NAME}.tar.gz ${ARCH_NAME}

#Moving archive to current directory
if [ "${CUR_PATH}" != "${PROJ_PATH}" ]
then
  mv ${ARCH_NAME}.tar.gz ${CUR_PATH}
fi

#Cleaning useless compiled temporary files
rm -r ${ARCH_NAME}
for file in ${TEMP_FILE[@]}
do
  rm ${file}
done

#PROFIT!
echo "complete"

