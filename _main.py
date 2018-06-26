# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import method
import time

def test(flag):
    print ("%10s%10s%15s%10s%10s%10s%20s%20s%20s%20s" % ("step","gamma","slow_rate","lamb","k","ratio","recall",'precision','coverage','popularity'))

    stplist = [50]
    gamlist = [0.06,0.02,0.001]
    slralist = [0.9]
    lamblist = [0.025,0.01]
    klist = [10,30,60,150]
    ratlist = [5,10,15]

    for steps in stplist:
        for gamma in gamlist:
            for slow_rate in slralist:
                for lamb in lamblist:
                    for kl in klist:
                        for ratio in ratlist:

                            test_count = 5
                            result_num = 2 if flag else 4
                            ans = [0] * result_num
                            for k in xrange(1, test_count + 1):
                                method.read_data(flag, k)

                                method.generate_matrix(flag,steps,gamma,slow_rate,lamb,kl,ratio)

                                if flag:
                                    b = method.evaluate_flag(flag)
                                else:
                                    b = method.evaluate_notflag(flag)

                                for x in xrange(result_num):
                                    ans[x] += b[x]

                            for x in xrange(result_num):
                                ans[x] /= test_count

                            print ("%9d%11.3f%12.2f%14.3f%9d%8d%19.3f%%%19.3f%%%19.3f%%%20.3f" % (steps,gamma,slow_rate,lamb,kl,ratio,ans[0] * 100,ans[1] * 100,ans[2] * 100,ans[3]))

if __name__ == '__main__':
    start = time.clock()
    # 0 打不打分
    # 1 打多少分
    test(0)
    end = time.clock()
    print ('finish all in %s' % str(end - start))
