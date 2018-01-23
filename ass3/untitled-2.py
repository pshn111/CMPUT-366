import matplotlib.pyplot as plt
def get_a(s):
    return min(s,100-s)
def value_iteration(S,theta,P):
    #list for all value of state
    V = [0 for i in range(101)]
    optimal = [0 for i in range(100)]
    loop = True
    while loop:
        target = 0
        for s in S:
            v = V[s]
            #list for all possible action
            A = [i+1 for i in range(get_a(s))]
            value = [0 for i in range(len(A)+1)]
            for a in A:
                if a+s>=100:
                    #print "%d %d %d %d"%(len(A),a,s+a,s-a)
                    value[a] = P*(1+V[s+a])+(1-P)*(0+V[s-a])
                else:
                    value[a] = P*V[s+a]+(1-P)*V[s-a]
            #find the max value except V[0] and V[100]
            V[s] = max(value)
            optimal[s-1] = value.index(V[s])
            target = max(target,abs(v-V[s]))
        if target<theta:
            loop = False
            V[100] = 1 
        
    return(V,optimal)
                
             
                


if __name__=="__main__":
    #array for all the state
    S = [i for i in range(1,100)]
    theta = 0.001
    #probability of get head
    P = 0.45
    (V,optimal)=value_iteration(S,theta,P)
    #show two graph
    plt.plot(V)
    plt.xlim([0,100])
    plt.xticks([1,25,50,75,99])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend()
    plt.show()
    
    plt.plot(optimal)
    plt.xlim([0,100])
    plt.xticks([1,25,50,75,99])
    plt.xlabel('Capital')
    plt.ylabel('Final policy')
    plt.legend()
    plt.show()    
    
