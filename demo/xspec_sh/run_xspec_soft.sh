# model 4 : soft

src_name=$1
xcm_name=$2
log_name=$3

model="powerlaw"
PC_command="newpar 1 3"
rm -rf $xcm_name
rm -rf $log_name
(
    echo data $src_name
    echo query yes
    echo ignore 0.-.3 1.-**
    echo statistic pgstat
    echo model $model
    echo 
    echo 
    echo 
    echo $PC_command
    echo fit
    echo 
    echo error 1.0 1
    echo 
    echo save all $xcm_name
    echo 
    echo exit
) | xspec > $log_name