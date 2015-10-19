
using Toms566

x = 0
function ff(x) 
    (x-3)^2 + 5
end
function gg(x) 
    2*(x-3)
end
H = 2 #here we have positive constant hessian so don't need to alter

function test(x0, optTol = 1e-6)
    x = x0
    f = ff(x)
    itn = 0
    while abs(gg(x))>optTol*abs(gg(x0))
        d = -gg(x)/2
        x = x+d
        f = [f;ff(x)]
        itn = itn+1
    end
    [itn; x; ff(x)]
end

test(0, 1e-6)

function NWTNsimple(p::Toms566.Problem, optTol, maxItn)
    x = p.x0
    f = p.obj(x)
    g = p.grd(x)
    H = p.hes(x)
    itn = 0
    
    while norm(p.grd(x))> optTol*norm(p.grd(p.x0))
        #first compute d ; the descent direction
        #we use hessian no matter if it is positive definite or not
        d = p.hes(x)\(-p.grd(x))
        x = x + d #newton's method alpha equals 1
        f = [f;p.obj(x)]
        itn = itn+1
        if itn == maxItn
            break
        end
    end
    [itn; p.obj(p.x0); p.obj(x)]
end

for i in 1:18
    p = Problem(i)
    @printf("%3i %3i %12.2e %12.2e\n", i,NWTNsimple(p, 1e-6, 500)...)
end
We see from the result that number 2, 7, 10, 11, 12, 14, 15, 18 does not seem to have converged. Also, since this is a simple Newton's method, we only know that those that converged have found the stationary points, but that we can't guarantee it is a minimizing point. So, now we inclue the modification of the Hessian matrix algorithm where we decompose the Hessian and alter the eigenvalues to be maximum of 1 and the eigenvalues. Also, we include a backtrack linesearch algorithm to find the stepsize.
#Modified Newton's(modifying the Hessian matrix) method with backtracking
function NWTNmod2bt(p::Toms566.Problem, optTol, eps, maxItn)
    x = p.x0
    f = p.obj(x)
    g = p.grd(x)
    itn = 0
    while norm(p.grd(x))>optTol*norm(p.grd(p.x0))
        #first compute d ; the descent direction
        #first we check if the hessian is positive definite
        if minimum(eig(p.hes(x))[1])<=0
            D, V = eig(p.hes(x))
            y = map(x->max(x,eps), D)
            H = V*Diagonal(y)*V'
            #eye(size(p.hes(x))[1], size(p.hes(x))[2])
        else 
            H = p.hes(x)
        end
        #now H is positive definite
        d = H\(-p.grd(x))
        #will do backtrack on alpha
        alpha = 1
        while p.obj(x+alpha*d)>p.obj(x) + alpha*0.4*dot(d,p.grd(x))
                alpha = alpha/2
            end
        x = x + alpha*d
        itn = itn+1
        if itn == maxItn 
            break 
        end
    end
    [itn; p.obj(p.x0); p.obj(x)]
end

for i in 1:18
    p = Problem(i)
    @printf("%3i %3i %12.2e %12.2e\n", i, NWTNmod2bt(p, 1e-6, 1, 100)...)
end

function NWTNmod1bt(p::Toms566.Problem, optTol, eps, mu, maxItn)
    x = p.x0
    f = p.obj(x)
    g = p.grd(x)
    itn = 0
    while norm(p.grd(x))>optTol*norm(p.grd(p.x0))
        #first compute d ; the descent direction
        #first we check if the hessian is positive definite
        if minimum(eig(p.hes(x))[1])<=0
            D, V = eig(p.hes(x))
            y = map(x->max(abs(x),eps), D)
            H = V*Diagonal(y)*V'
            #eye(size(p.hes(x))[1], size(p.hes(x))[2])
        else 
            H = p.hes(x)
        end
        #now H is positive definite
        d = H\(-p.grd(x))
        #will do backtrack on alpha
        alpha = 1
        while p.obj(x+alpha*d)>p.obj(x) + alpha*mu*dot(d,p.grd(x))
                alpha = alpha/2
            end
        x = x + alpha*d
        itn = itn+1
        if itn == maxItn 
            break 
        end
    end
    [itn; p.obj(p.x0); p.obj(x)]
end

for i in [2;7;11:15;18]
    p = Problem(i)
    @printf("%3i %3i %12.2e %12.2e\n", i, NWTNmod1bt(p, 1e-6, 1, 1e-4, 100)...)
end

#Quasi Newton method
function NWTNquasi(p::Toms566.Problem, optTol, maxItn)
    itn = 0
    x = p.x0
    D, V = eig(p.hes(p.x0))
    y = map(x->max(x,1), D)
    H = V*Diagonal(1./y)*V'#inversion of modified hessian matrix
    while norm(p.grd(x))>optTol*norm(p.grd(p.x0))
        #first compute d ; the descent direction
        #first we check if the hessian is positive definite
        d = H*(-p.grd(x))
        
    #finding alpha satifying Wolfe’s condition
        alpha = 1
        while p.obj(x+alpha*d)>p.obj(x) + alpha*1e-3*dot(d,p.grd(x))
            alpha = alpha*(1/2)
            if alpha <= 1e-6
                break
            end
        end
        while dot(d,p.grd(x+alpha*d))<dot(0.9*d,p.grd(x))
            alpha = alpha*(1/2)
            if alpha <= 1e-6
                break
            end
        end

        x = x+alpha*d
        s = alpha*d
        y = p.grd(x)-p.grd(x-s)
        ro = 1/dot(y,s)
        H = (eye(H)-ro*s*y')*H*(eye(H)-ro*y*s') + ro*s*s'
        itn = itn + 1
        if itn == maxItn 
            break 
        end
    end
    [itn; p.obj(p.x0); p.obj(x)]
end

for i in 1:18
    p = Problem(i)
    @printf("%3i %3i %12.2e %12.2e\n", i,NWTNquasi(p, 1e-8, 1000)...)
end

Admissions = readdlm("/Users/Irene/Documents/UCD/2015fall/MAT258a/hw2/binary.csv", ',', header=true);

y = Admissions[1][1:400,1];#response variable 400x1
u0 = Admissions[1][1:400,2:3]*float(Diagonal([1/400, 1]));#GRE and GPA
u = (cat(2,[1 for i = 1:400],u0))';#the vector(data) that we do inner product to the parameters beta and a 400x3

#define the objective, gradient and hessian functions
#objective function
function f(x)#x is the parameter values and f is the negative log likelihood 
    obj = 0
    for i = 1:400
        obj = obj - y[i,]*(x'*u[:,i]) + log(1+exp(x'*u[:,i]))
    end
    obj
end

function gf(x)#compute the graident
    g = [0;0;0]
    for i = 1:400
        g = g - vec(y[i,].*u[:,i] - ( (exp(x'*u[:,i])) / (1+exp(x'*u[:,i])) ).*u[:,i])  
    end
    g
end

function hf(x)#compute the hessian
    h = 0
    for i = 1:400
        h = h + (exp(x'*u[:,i]))/(1 + exp(x'*u[:,i])).*u[:,i]*u[:,i]'
    end
    h
end

#modified newton's method(hessian alteration) with backtracking alpha
function NWTN(x0, optTol, eps, mu, maxItn)
    x = x0
    itn = 0
    for itn = 1:maxItn
        
        if norm(gf(x))<=optTol*norm(gf(x0))
            break
        end
        
        #to decide the step direction, first alter hessian as needed
        D, V = eig(hf(x))

        if minimum(D)<=0
            pD = map(x->max(x,eps), D)
            h = V*pD*V'
        else
            h = hf(x)
        end

        d = -h\gf(x)#we found the descending direction!
        #now we need to choose alpha
        alpha = 1
        while ( f(x+alpha*d) - f(x) + alpha*mu*(d'*gf(x)) )[]>0
            alpha = (1/2)*alpha
            if alpha<1e-7
                break
            end
        end
        x = x+alpha*d
        itn = itn + 1 
    end
    [itn;x;f(x)]
end

x0 = [0;0;0]
NWTN(x0, 1e-6, 1, 0.2, 1000)


