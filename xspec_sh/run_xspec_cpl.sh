# model 1 : cutoffpl

grp_name=$1
xcm_name=$2
log_name=$3

model="tbabs*(cflux*powerlaw+cflux*cutoffpl+gauss)"
tbabs_command="newpar 1 1e-2"
PC_command="newpar 4 -12;newpar 5 2;newpar 6 1 -1"
SE_command="newpar 9 -13;newpar 10 4 0.1 -3 -3 10 10;newpar 11 10 0.1 0.01 0.01 500 500;newpar 12 1 -1"
gauss_command="newpar 13 6.4 -1;newpar 14 0.019 0.1 0 0 1 1;newpar 15 1e-6"
rm -rf $xcm_name
rm -rf $log_name
(
    echo data $grp_name
    echo query yes
    echo ignore 0.-.3 8.-**
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
    echo $gauss_command
    echo fit
    echo 
    echo step 11 0.1 10 100
    echo 
    echo freeze 1 5 10 11 14 15
    echo newpar 3 2
    echo newpar 8 2
    echo fit
    echo 
    echo addcomp 4 const
    echo 
    echo newpar 7 0.5
    echo newpar 10=4
    echo fit
    echo 
    echo thaw 1 5 11 12 15 16
    echo fit
    echo 
    echo error 1.0 1
    echo 
    echo fit
    echo 
    echo error 1.0 5
    echo 
    echo fit
    echo 
    echo error 1.0 7
    echo 
    echo fit
    echo 
    echo error 1.0 11
    echo 
    echo fit
    echo 
    echo error 1.0 12 
    echo 
    echo fit
    echo 
    echo error 1.0 15
    echo 
    echo eqwidth 7 err 400 68
    echo 
    echo save all $xcm_name
    echo 
    echo exit
) | xspec > $log_name