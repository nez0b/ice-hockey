
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    #input
    to_input( './kart.png' ),

    to_ConvConvRelu( name='b1', s_filer=' ', n_filer=(16,16), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption=' Block 1'  ),
    to_ConvConvRelu( name='b2', s_filer=' ', n_filer=(32,32), offset="(2.,0,0)", to="(0,0,0)", width=(3,3), height=30, depth=30, caption=' Block 1'),
    #to_connection("b1", "b2"),
    to_ConvConvRelu( name='b3', s_filer=' ', n_filer=(64,64), offset="(4.,0,0)", to="(0,0,0)", width=(4,4), height=20, depth=20, caption=' Block 1'),
    to_ConvConvRelu( name='b4', s_filer=' ', n_filer=(128,128), offset="(6.,0,0)", to="(0,0,0)", width=(5,5), height=10, depth=10, caption=' Block 1'),

    to_ConvConvRelu( name='ub4', s_filer=' ', n_filer=(128,128), offset="(10.,0,0)", to="(0,0,0)", width=(5,5), height=10, depth=10, caption=' UpBlock 4'),
    to_ConvConvRelu( name='ub3', s_filer=' ', n_filer=(64,64), offset="(13.,0,0)", to="(0,0,0)", width=(4,4), height=20, depth=20, caption=' UpBlock 3'),
    to_ConvConvRelu( name='ub2', s_filer=' ', n_filer=(32,32), offset="(16.,0,0)", to="(0,0,0)", width=(3,3), height=30, depth=30, caption=' UpBlock 2'),
    to_ConvConvRelu( name='ub1', s_filer=' ', n_filer=(16,16), offset="(18.,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption=' UpBlock 1'),

    to_connection("b4", "ub4"),
    to_skip( of='b1', to='ub1', pos=1.25),
    to_skip( of='b2', to='ub2', pos=1.25),
    to_skip( of='b3', to='ub3', pos=1.25),


    #block_2ConvPool(name='b2', botton='pool_b1', top='pool_b2', s_filer=64, n_filer=32, offset="(0,0,0)", size=(32,32,2.5), opacity=0.5 ),
    #to_ConvRes("convres1", s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=6, height=40, depth=40, opacity=0.2, caption=" " ),
    #to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
    #to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
    #to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    #to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(conv1-east)", height=32, depth=32, width=2 ),
    #to_connection( "pool1", "conv2"), 
    #to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    #to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    #to_connection("pool2", "soft1"),    
    #to_Sum("sum1", offset="(1.5,0,0)", to="(soft1-east)", radius=2.5, opacity=0.6),
    #to_connection("soft1", "sum1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
