# model 3 : powerlaw

src_name=$1
xcm_name=$2
log_name=$3

model="tbabs*(cflux*powerlaw+cflux*powerlaw+gauss)"
tbabs_command="newpar 1 1e-2"
PC_command="newpar 4 -12;newpar 5 2;newpar 6 1 -1"
SE_command="newpar 9 -13;newpar 10 4;newpar 11 1 -1"
GA_command="newpar 12 6.4 -1;newpar 13 0.019 0.1 0 0 1 1;newpar 14 1e-6"
rm -rf $xcm_name
rm -rf $log_name
(
    echo data $src_name
    echo query yes
    echo ignore 0.-.3 8.-**
    echo statistic pgstat
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
    echo 
    echo 
    echo 
    echo 
    echo 
    echo $tbabs_command
    echo $PC_command
    echo $SE_command
    echo $GA_command
    echo fit
    echo 
    echo error 1.0 1
    echo 
    echo fit
    echo 
    echo error 1.0 4
    echo 
    echo fit
    echo 
    echo error 1.0 5
    echo 
    echo fit
    echo 
    echo error 1.0 9
    echo 
    echo fit
    echo 
    echo error 1.0 10
    echo 
    echo fit
    echo 
    echo error 1.0 14
    echo 
    echo save all $xcm_name
    echo 
    echo exit
) | xspec > $log_name