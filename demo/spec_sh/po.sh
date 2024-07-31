z=$1
lum=$2
gamma=$3
seed=$4
spec_dir=$5
pha_file=$6
arf_file=$7
out_pre=$8
out_dir=$9
src_expo=${10}
bkg_expo=${11}

fkpha_file=$out_pre".fak"
fkbkg_file=$out_pre"_bkg.fak"
fkarf_file=$out_pre".arf"

cwd=$pwd

cd $spec_dir

model="clumin*zpowerlw"
(
    echo model $model
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo newpar 3 $z
    echo newpar 4 $lum
    echo newpar 5 $gamma
    echo newpar 6 $z
    echo newpar 7 1 -1
    echo 
    echo show
    echo 
    echo data $pha_file
    echo 
    echo xset seed $seed 
    echo 
    echo fakeit
    echo 
    echo 
    echo $fkpha_file
    echo $src_expo,1,$bkg_expo
    echo 
    echo exit
) | xspec

mv $fkpha_file $fkbkg_file $out_dir
cp $arf_file $out_dir/$fkarf_file

cd $cwd
