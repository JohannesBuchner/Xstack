# model 5 : fixed shape

grp_name=$1
xcm_name=$2
log_name=$3

model="cflux*powerlaw+const*cflux*bbody"
PC_command="newpar 2 2;newpar 4 2 -1;newpar 5 1 -1"
SE_command="newpar 6 0.1;newpar 8 2;newpar 9=3;newpar 10 0.1 -1;newpar 11 1 -1"
rm -rf $xcm_name
rm -rf $log_name
(
    echo data $grp_name
    echo query yes
    echo ignore 0.-.3 8.1-**
    echo model $model
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo 
    echo $PC_command
    echo $SE_command
    echo fit
    echo 
    echo error maximum 5 1.0 6
    echo 
    echo save all $xcm_name
    echo 
    echo exit
) | xspec > $log_name