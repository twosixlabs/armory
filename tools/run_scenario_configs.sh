# Outputs log files. Usage: ./tools/run_scenario_configs.sh
for FILE in ./scenario_configs/*json;
do
	echo "armory run $FILE --check"
	python -m armory run $FILE --check > $FILE.log 2>$FILE_error.log
done
