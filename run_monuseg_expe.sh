echo  "Rot Equiv 100 pct"

python3 -W ignore train_rot_equiv.py --config config/rot_equiv_config100.yaml

echo  "Rot Equiv 10 pct"


python3 -W ignore train_rot_equiv.py --config config/rot_equiv_config10.yaml

echo  "Rot Equiv 5 pct"

python3 -W ignore train_rot_equiv.py --config config/rot_equiv_config5.yaml


echo "Fully_sup  100%"
python3 -W ignore train_fully_supervised.py --config config/fully_sup_config100.yaml

echo "Fully_sup  100% rot*"
python3 -W ignore train_fully_supervised.py --config config/fully_sup_config100rot.yaml

echo "Fully_sup  10%"
python3 -W ignore train_fully_supervised.py --config config/fully_sup_config10.yaml

echo "Fully_sup  10% rot*"
python3 -W ignore train_fully_supervised.py --config config/fully_sup_config10rot.yaml

echo "Fully_sup  5%"
python3 -W ignore train_fully_supervised.py --config config/fully_sup_config5.yaml

echo "Fully_sup  5% rot*"
python3 -W ignore train_fully_supervised.py --config config/fully_sup_config5rot.yaml