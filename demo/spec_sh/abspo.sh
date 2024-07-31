nh=$1
z=$2
lum=$3
gamma=$4
seed=$5
spec_dir=$6
pha_file=$7
arf_file=$8
out_pre=$9
out_dir=${10}
src_expo=${11}
bkg_expo=${12}

fkpha_file=$out_pre".fak"
fkbkg_file=$out_pre"_bkg.fak"
fkarf_file=$out_pre".arf"

cwd=$pwd

cd $spec_dir

model="tbabs*(clumin*zpowerlw)"
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
    echo newpar 1 $nh
    echo newpar 4 $z
    echo newpar 5 $lum
    echo newpar 6 $gamma
    echo newpar 7 $z
    echo newpar 8 1 -1
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
