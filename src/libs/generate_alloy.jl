using DataStructures
using LinearAlgebra
using NearestNeighbors
using SparseArrays
using Interpolations
include("../materials/materials.jl")


function Deque_to_vec(Elist,Edict,vararg...;Transform=nothing)
    NN=length(Elist)
    R=zeros(Float64,NN,3)
    if isnothing(Transform)
        for Ri in Elist
            idx=Edict[Ri]
            R[idx,:]=Ri
        end
        return R
    else
        for Ri in Elist
            idx=Edict[Ri]
            R[idx,:]=Transform(Ri,vararg)
        end
    return R
    end
end


function explore_alloy(nnlist::Array{Int64,2},in_shape::Function,vararg...;start_pos::Array{Int64,1}=[0,0,0])
    #=
    nnlist: Array{Int64,3,NN} the Neighbouring direction, where NN is the number of neighbouring directions
    in_shape: ans=in_shape(R,vararg) a function that determine whether a given position R is in shape or not, vararg are 
    input variables
    start_point: a vector that defines the starting position
    =#
    let 
    Udict=Dict{Vector{Int64},Int64}()
    Edict=Dict{Vector{Int64},Int64}()
    Edictinv=Dict{Int64,Vector{Int64}}()
    
    
    Ulist=Deque{Vector{Int64}}()
    Elist=Deque{Vector{Int64}}()
    idx=1
    Udict[start_pos]=idx
    push!(Ulist,start_pos)
    while (length(Ulist)>0)
        Ri=popfirst!(Ulist)
        idx2=pop!(Udict,Ri)
        Edict[Ri]=idx2
        Edictinv[idx2]=Ri
        push!(Elist,Ri)
        for j=1:12
            Rij=nnlist[:,j]
            Rj=Ri+Rij 
            exists=haskey(Udict,Rj)
            exists= exists|| haskey(Edict,Rj)
            if ((!exists) && in_shape(Rj,vararg))
                idx=idx+1
                Udict[Rj]=idx 
                push!(Ulist,Rj)
            end 
        end 
    end

    return Elist,Edict
    end
end 


function Au_getIntraHopping(Rij)
    if norm(Rij)<1e-6 
        return Diagonal([-7.069,-5.4325,-5.4325,-5.4325,-5.4325,-5.4325,0.9004,0.9004,0.9004])
    end
    nij=Rij./norm(Rij)
    dij=norm(Rij)

    
    
    function getHoppingParameters(d)
        d0=2.8837
        I0_ss_sig=0.5000
        p_ss_sig=3.4130
        ss_sig=I0_ss_sig*exp(-p_ss_sig*(d/d0-1))
        I0_sp_sig=47.4232
        p_sp_sig=75.4943
        sp_sig=I0_sp_sig*exp(-p_sp_sig*(d/d0-1))
        I0_sd_sig=0.0339
        p_sd_sig=1.8747
        sd_sig=I0_sd_sig*exp(-p_sd_sig*(d/d0-1))
        I0_pp_sig=1.0660
        p_pp_sig=4.5685
        pp_sig=I0_pp_sig*exp(-p_pp_sig*(d/d0-1))
        I0_pp_pi=0.1271
        p_pp_pi=6.1124
        pp_pi=I0_pp_pi*exp(-p_pp_pi*(d/d0-1))
        I0_pd_sig=99.8976 
        p_pd_sig=99.9935
        pd_sig=I0_pd_sig*exp(-p_pd_sig*(d/d0-1))
        I0_pd_pi=-98.6434
        p_pd_pi=91.1682
        pd_pi=I0_pd_pi*exp(-p_pd_pi*(d/d0-1))
        I0_dd_sig=0.9232
        p_dd_sig=1.8897
        dd_sig=I0_dd_sig*exp(-p_dd_sig*(d/d0-1))
        I0_dd_pi=0.3693
        p_dd_pi=2.0762
        dd_pi=I0_dd_pi*exp(-p_dd_pi*(d/d0-1))
        I0_dd_del=-0.9722 
        p_dd_del=1.3285 
        dd_del=I0_dd_del*exp(-p_dd_del*(d/d0-1))
        return sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del
        #return 0,0,0,0,0,0,0,0,0,0
    end
    l,m,n=nij
    sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del=getHoppingParameters(dij)
    V=zeros(ComplexF64,9,9)
    #Same orbital interaction
    V[1,1]=ss_sig
    V[2,2]=( 3*l^2*m^2*dd_sig + ( l^2 + m^2 - 4*l^2*m^2)*dd_pi +(n^2+l^2*m^2)*dd_del)
    V[3,3]=( 3*l^2*n^2*dd_sig + ( l^2 + n^2 - 4*l^2*n^2)*dd_pi +(m^2+l^2*n^2)*dd_del)
    V[4,4]=( 3*m^2*n^2*dd_sig + ( m^2 + n^2 - 4*m^2*n^2)*dd_pi +(l^2+m^2*n^2)*dd_del)
    V[5,5]=(3/4*(l^2-m^2)^2*dd_sig+(l^2+m^2-(l^2-m^2)^2)*dd_pi+(n^2+(l^2-m^2)^2/4)*dd_del)
    V[6,6]=((n^2-.5*(l^2+m^2))^2*dd_sig+3*n^2*(l^2+m^2)*dd_pi +3/4*(l^2+m^2)^2*dd_del)
    V[7,7]=(l^2*pp_sig+(1-l^2)*pp_pi)
    V[8,8]=(m^2*pp_sig+(1-m^2)*pp_pi)
    V[9,9]=(n^2*pp_sig+(1-n^2)*pp_pi)
    #p-p interaction
    V[7,8]=(l*m*pp_sig-l*m*pp_pi)
    V[7,9]=(l*n*pp_sig-l*n*pp_pi)
    V[8,9]=(m*n*pp_sig-m*n*pp_pi)
    #dd interaction
    V[2,3]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[2,4]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[2,5]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[2,6]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[3,4]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[3,5]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[3,6]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[4,5]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[4,6]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[5,6]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #s-d interaction
    V[1,2]=3^.5 * l*m*sd_sig
    V[1,3]=3^.5*l*n*sd_sig
    V[1,4]=3^.5*n*m*sd_sig
    V[1,5]=3^.5/2*(l^2-m^2)*sd_sig
    V[1,6]=(n^2-.5*(l^2+m^2))*sd_sig
    #sp interaction
    V[1,7]=(l*sp_sig)
    V[1,8]=(m*sp_sig)
    V[1,9]=(n*sp_sig)
    #pd interaction
    V[7,2]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[8,2]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[9,2]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[7,3]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[8,3]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[9,3]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[7,4]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[8,4]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[9,4]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[7,5]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[8,5]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[9,5]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[7,6]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[8,6]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[9,6]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    #For those hopping not listed in SK table
    #Compute the opposite hopping and use Hermiticity of Hamiltonian to 
    #derive these elements. 
    #Now flip the direction:
    #l,m,n=nij.*-1
    #dd interaction
    V[3,2]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[4,2]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[5,2]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[6,2]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[4,3]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[5,3]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[6,3]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[5,4]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[6,4]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[6,5]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #p-p interaction
    V[8,7]=(l*m*pp_sig-l*m*pp_pi)
    V[9,7]=(l*n*pp_sig-l*n*pp_pi)
    V[9,8]=(m*n*pp_sig-m*n*pp_pi)
    #ds interaction
    V[2,1]=3^.5 * l*m*sd_sig
    V[3,1]=3^.5*l*n*sd_sig
    V[4,1]=3^.5*n*m*sd_sig  
    V[5,1]=3^.5/2*(l^2-m^2)*sd_sig
    V[6,1]=(n^2-.5*(l^2+m^2))*sd_sig
    #ps interaction
    V[7,1]=(l*sp_sig)
    V[8,1]=(m*sp_sig)
    V[9,1]=(n*sp_sig)
    #dp interaction
    V[2,7]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[2,8]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[2,9]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[3,7]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[3,8]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[3,9]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[4,7]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[4,8]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[4,9]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[5,7]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[5,8]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[5,9]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[6,7]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[6,8]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[6,9]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    return Diagonal(V)
    #return V

end 

function Au_getInterHopping(Rij)
    nij=Rij./norm(Rij)
    dij=norm(Rij)
    function getHoppingParameters(d)
        d0=2.8837
        V0_ss_sig=-0.9261
        q_ss_sig=3.4147
        ss_sig=V0_ss_sig*exp(-q_ss_sig*(d/d0-1))
        V0_sp_sig=1.3669
        q_sp_sig=3.0152
        sp_sig=V0_sp_sig*exp(-q_sp_sig*(d/d0-1))
        V0_sd_sig=-0.6941
        q_sd_sig=4.0517
        sd_sig=V0_sd_sig*exp(-q_sd_sig*(d/d0-1))
        V0_pp_sig=1.7926
        q_pp_sig=2.5772
        pp_sig=V0_pp_sig*exp(-q_pp_sig*(d/d0-1))
        V0_pp_pi=-0.5155
        q_pp_pi=2.9059
        pp_pi=V0_pp_pi*exp(-q_pp_pi*(d/d0-1))
        V0_pd_sig=-0.9479
        q_pd_sig=4.2849
        pd_sig=V0_pd_sig*exp(-q_pd_sig*(d/d0-1))
        V0_pd_pi=0.2972
        q_pd_pi=4.6552
        pd_pi=V0_pd_pi*exp(-q_pd_pi*(d/d0-1))
        V0_dd_sig=-0.6844
        q_dd_sig=5.4403
        dd_sig=V0_dd_sig*exp(-q_dd_sig*(d/d0-1))
        V0_dd_pi=0.3381
        q_dd_pi=5.0338
        dd_pi=V0_dd_pi*exp(-q_dd_pi*(d/d0-1))
        V0_dd_del=-0.0592
        q_dd_del=2.4849
        dd_del=V0_dd_del*exp(-q_dd_del*(d/d0-1))
        return sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del
    end 
    l,m,n=nij
    sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del=getHoppingParameters(dij)
    V=zeros(ComplexF64,9,9)
    #Same orbital interaction
    V[1,1]=ss_sig
    V[2,2]=( 3*l^2*m^2*dd_sig + ( l^2 + m^2 - 4*l^2*m^2)*dd_pi +(n^2+l^2*m^2)*dd_del)
    V[3,3]=( 3*l^2*n^2*dd_sig + ( l^2 + n^2 - 4*l^2*n^2)*dd_pi +(m^2+l^2*n^2)*dd_del)
    V[4,4]=( 3*m^2*n^2*dd_sig + ( m^2 + n^2 - 4*m^2*n^2)*dd_pi +(l^2+m^2*n^2)*dd_del)
    V[5,5]=(3/4*(l^2-m^2)^2*dd_sig+(l^2+m^2-(l^2-m^2)^2)*dd_pi+(n^2+(l^2-m^2)^2/4)*dd_del)
    V[6,6]=((n^2-.5*(l^2+m^2))^2*dd_sig+3*n^2*(l^2+m^2)*dd_pi +3/4*(l^2+m^2)^2*dd_del)
    V[7,7]=(l^2*pp_sig+(1-l^2)*pp_pi)
    V[8,8]=(m^2*pp_sig+(1-m^2)*pp_pi)
    V[9,9]=(n^2*pp_sig+(1-n^2)*pp_pi)
    #p-p interaction
    V[7,8]=(l*m*pp_sig-l*m*pp_pi)
    V[7,9]=(l*n*pp_sig-l*n*pp_pi)
    V[8,9]=(m*n*pp_sig-m*n*pp_pi)
    #dd interaction
    V[2,3]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[2,4]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[2,5]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[2,6]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[3,4]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[3,5]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[3,6]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[4,5]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[4,6]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[5,6]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #s-d interaction
    V[1,2]=3^.5 * l*m*sd_sig
    V[1,3]=3^.5*l*n*sd_sig
    V[1,4]=3^.5*n*m*sd_sig
    V[1,5]=3^.5/2*(l^2-m^2)*sd_sig
    V[1,6]=(n^2-.5*(l^2+m^2))*sd_sig
    #sp interaction
    V[1,7]=(l*sp_sig)
    V[1,8]=(m*sp_sig)
    V[1,9]=(n*sp_sig)
    #pd interaction
    V[7,2]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[8,2]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[9,2]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[7,3]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[8,3]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[9,3]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[7,4]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[8,4]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[9,4]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[7,5]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[8,5]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[9,5]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[7,6]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[8,6]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[9,6]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    #For those hopping not listed in SK table
    #Compute the opposite hopping and use Hermiticity of Hamiltonian to 
    #derive these elements. 
    #Now flip the direction:
    l,m,n=nij.*-1
    #dd interaction
    V[3,2]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[4,2]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[5,2]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[6,2]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[4,3]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[5,3]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[6,3]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[5,4]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[6,4]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[6,5]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #p-p interaction
    V[8,7]=(l*m*pp_sig-l*m*pp_pi)
    V[9,7]=(l*n*pp_sig-l*n*pp_pi)
    V[9,8]=(m*n*pp_sig-m*n*pp_pi)
    #ds interaction
    V[2,1]=3^.5 * l*m*sd_sig
    V[3,1]=3^.5*l*n*sd_sig
    V[4,1]=3^.5*n*m*sd_sig  
    V[5,1]=3^.5/2*(l^2-m^2)*sd_sig
    V[6,1]=(n^2-.5*(l^2+m^2))*sd_sig
    #ps interaction
    V[7,1]=(l*sp_sig)
    V[8,1]=(m*sp_sig)
    V[9,1]=(n*sp_sig)
    #dp interaction
    V[2,7]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[2,8]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[2,9]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[3,7]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[3,8]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[3,9]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[4,7]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[4,8]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[4,9]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[5,7]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[5,8]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[5,9]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[6,7]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[6,8]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[6,9]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    return V
end 

function Ag_getIntraHopping(Rij)
    if norm(Rij)<1e-6 
        return Diagonal([-3.5253,-5.5780,-5.5780,-5.5780,-5.5780,-5.5780,1.4098,1.4098,1.4098])
    end
    nij=Rij./norm(Rij)
    dij=norm(Rij)



    function getHoppingParameters(d)
        d0=2.8890
        I0_ss_sig=0.2888
        p_ss_sig=3.3967
        ss_sig=I0_ss_sig*exp(-p_ss_sig*(d/d0-1))
        I0_sp_sig=27.5677
        p_sp_sig=99.1027
        sp_sig=I0_sp_sig*exp(-p_sp_sig*(d/d0-1))
        I0_sd_sig=-0.5037
        p_sd_sig=0.4269
        sd_sig=I0_sd_sig*exp(-p_sd_sig*(d/d0-1))
        I0_pp_sig=0.3040
        p_pp_sig=3.1565
        pp_sig=I0_pp_sig*exp(-p_pp_sig*(d/d0-1))
        I0_pp_pi=0.2953
        p_pp_pi=5.1796
        pp_pi=I0_pp_pi*exp(-p_pp_pi*(d/d0-1))
        I0_pd_sig=99.9958 
        p_pd_sig=99.8750
        pd_sig=I0_pd_sig*exp(-p_pd_sig*(d/d0-1))
        I0_pd_pi=-99.1472
        p_pd_pi=99.1178
        pd_pi=I0_pd_pi*exp(-p_pd_pi*(d/d0-1))
        I0_dd_sig=0.7252
        p_dd_sig=2.3977
        dd_sig=I0_dd_sig*exp(-p_dd_sig*(d/d0-1))
        I0_dd_pi=0.2511
        p_dd_pi=2.1463
        dd_pi=I0_dd_pi*exp(-p_dd_pi*(d/d0-1))
        I0_dd_del=-0.9280
        p_dd_del=1.7029 
        dd_del=I0_dd_del*exp(-p_dd_del*(d/d0-1))
        return sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del
    end
    l,m,n=nij
    sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del=getHoppingParameters(dij)
    V=zeros(ComplexF64,9,9)
    #Same orbital interaction
    V[1,1]=ss_sig
    V[2,2]=( 3*l^2*m^2*dd_sig + ( l^2 + m^2 - 4*l^2*m^2)*dd_pi +(n^2+l^2*m^2)*dd_del)
    V[3,3]=( 3*l^2*n^2*dd_sig + ( l^2 + n^2 - 4*l^2*n^2)*dd_pi +(m^2+l^2*n^2)*dd_del)
    V[4,4]=( 3*m^2*n^2*dd_sig + ( m^2 + n^2 - 4*m^2*n^2)*dd_pi +(l^2+m^2*n^2)*dd_del)
    V[5,5]=(3/4*(l^2-m^2)^2*dd_sig+(l^2+m^2-(l^2-m^2)^2)*dd_pi+(n^2+(l^2-m^2)^2/4)*dd_del)
    V[6,6]=((n^2-.5*(l^2+m^2))^2*dd_sig+3*n^2*(l^2+m^2)*dd_pi +3/4*(l^2+m^2)^2*dd_del)
    V[7,7]=(l^2*pp_sig+(1-l^2)*pp_pi)
    V[8,8]=(m^2*pp_sig+(1-m^2)*pp_pi)
    V[9,9]=(n^2*pp_sig+(1-n^2)*pp_pi)
    #p-p interaction
    V[7,8]=(l*m*pp_sig-l*m*pp_pi)
    V[7,9]=(l*n*pp_sig-l*n*pp_pi)
    V[8,9]=(m*n*pp_sig-m*n*pp_pi)
    #dd interaction
    V[2,3]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[2,4]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[2,5]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[2,6]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[3,4]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[3,5]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[3,6]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[4,5]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[4,6]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[5,6]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #s-d interaction
    V[1,2]=3^.5 * l*m*sd_sig
    V[1,3]=3^.5*l*n*sd_sig
    V[1,4]=3^.5*n*m*sd_sig
    V[1,5]=3^.5/2*(l^2-m^2)*sd_sig
    V[1,6]=(n^2-.5*(l^2+m^2))*sd_sig
    #sp interaction
    V[1,7]=(l*sp_sig)
    V[1,8]=(m*sp_sig)
    V[1,9]=(n*sp_sig)
    #pd interaction
    V[7,2]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[8,2]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[9,2]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[7,3]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[8,3]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[9,3]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[7,4]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[8,4]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[9,4]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[7,5]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[8,5]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[9,5]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[7,6]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[8,6]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[9,6]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    #For those hopping not listed in SK table
    #Compute the opposite hopping and use Hermiticity of Hamiltonian to 
    #derive these elements. 
    #Now flip the direction:
    #l,m,n=nij.*-1
    #dd interaction
    V[3,2]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[4,2]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[5,2]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[6,2]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[4,3]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[5,3]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[6,3]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[5,4]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[6,4]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[6,5]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #p-p interaction
    V[8,7]=(l*m*pp_sig-l*m*pp_pi)
    V[9,7]=(l*n*pp_sig-l*n*pp_pi)
    V[9,8]=(m*n*pp_sig-m*n*pp_pi)
    #ds interaction
    V[2,1]=3^.5 * l*m*sd_sig
    V[3,1]=3^.5*l*n*sd_sig
    V[4,1]=3^.5*n*m*sd_sig  
    V[5,1]=3^.5/2*(l^2-m^2)*sd_sig
    V[6,1]=(n^2-.5*(l^2+m^2))*sd_sig
    #ps interaction
    V[7,1]=(l*sp_sig)
    V[8,1]=(m*sp_sig)
    V[9,1]=(n*sp_sig)
    #dp interaction
    V[2,7]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[2,8]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[2,9]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[3,7]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[3,8]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[3,9]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[4,7]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[4,8]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[4,9]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[5,7]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[5,8]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[5,9]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[6,7]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[6,8]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[6,9]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    return Diagonal(V)
    #return V

end

function Ag_getInterHopping(Rij)
    nij=Rij./norm(Rij)
    dij=norm(Rij)
    function getHoppingParameters(d)
        d0=2.8890
        V0_ss_sig=-0.8864
        q_ss_sig=2.5004
        ss_sig=V0_ss_sig*exp(-q_ss_sig*(d/d0-1))
        V0_sp_sig=1.2238
        q_sp_sig=1.7035
        sp_sig=V0_sp_sig*exp(-q_sp_sig*(d/d0-1))
        V0_sd_sig=-0.5268
        q_sd_sig=3.8742
        sd_sig=V0_sd_sig*exp(-q_sd_sig*(d/d0-1))
        V0_pp_sig=1.5428
        q_pp_sig=1.4366
        pp_sig=V0_pp_sig*exp(-q_pp_sig*(d/d0-1))
        V0_pp_pi=-0.5098
        q_pp_pi=4.0872
        pp_pi=V0_pp_pi*exp(-q_pp_pi*(d/d0-1))
        V0_pd_sig=-0.6058
        q_pd_sig=5.3603
        pd_sig=V0_pd_sig*exp(-q_pd_sig*(d/d0-1))
        V0_pd_pi=0.1868
        q_pd_pi=6.7768
        pd_pi=V0_pd_pi*exp(-q_pd_pi*(d/d0-1))
        V0_dd_sig=-0.4540
        q_dd_sig=5.5990 
        dd_sig=V0_dd_sig*exp(-q_dd_sig*(d/d0-1))
        V0_dd_pi=0.2456
        q_dd_pi=5.2114 
        dd_pi=V0_dd_pi*exp(-q_dd_pi*(d/d0-1))
        V0_dd_del=-0.0496
        q_dd_del=3.9194
        dd_del=V0_dd_del*exp(-q_dd_del*(d/d0-1))
        return sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del
    end 
    l,m,n=nij
    sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del=getHoppingParameters(dij)
    V=zeros(ComplexF64,9,9)
    #Same orbital interaction
    V[1,1]=ss_sig
    V[2,2]=( 3*l^2*m^2*dd_sig + ( l^2 + m^2 - 4*l^2*m^2)*dd_pi +(n^2+l^2*m^2)*dd_del)
    V[3,3]=( 3*l^2*n^2*dd_sig + ( l^2 + n^2 - 4*l^2*n^2)*dd_pi +(m^2+l^2*n^2)*dd_del)
    V[4,4]=( 3*m^2*n^2*dd_sig + ( m^2 + n^2 - 4*m^2*n^2)*dd_pi +(l^2+m^2*n^2)*dd_del)
    V[5,5]=(3/4*(l^2-m^2)^2*dd_sig+(l^2+m^2-(l^2-m^2)^2)*dd_pi+(n^2+(l^2-m^2)^2/4)*dd_del)
    V[6,6]=((n^2-.5*(l^2+m^2))^2*dd_sig+3*n^2*(l^2+m^2)*dd_pi +3/4*(l^2+m^2)^2*dd_del)
    V[7,7]=(l^2*pp_sig+(1-l^2)*pp_pi)
    V[8,8]=(m^2*pp_sig+(1-m^2)*pp_pi)
    V[9,9]=(n^2*pp_sig+(1-n^2)*pp_pi)
    #p-p interaction
    V[7,8]=(l*m*pp_sig-l*m*pp_pi)
    V[7,9]=(l*n*pp_sig-l*n*pp_pi)
    V[8,9]=(m*n*pp_sig-m*n*pp_pi)
    #dd interaction
    V[2,3]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[2,4]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[2,5]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[2,6]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[3,4]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[3,5]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[3,6]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[4,5]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[4,6]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[5,6]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #s-d interaction
    V[1,2]=3^.5 * l*m*sd_sig
    V[1,3]=3^.5*l*n*sd_sig
    V[1,4]=3^.5*n*m*sd_sig
    V[1,5]=3^.5/2*(l^2-m^2)*sd_sig
    V[1,6]=(n^2-.5*(l^2+m^2))*sd_sig
    #sp interaction
    V[1,7]=(l*sp_sig)
    V[1,8]=(m*sp_sig)
    V[1,9]=(n*sp_sig)
    #pd interaction
    V[7,2]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[8,2]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[9,2]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[7,3]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[8,3]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[9,3]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[7,4]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[8,4]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[9,4]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[7,5]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[8,5]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[9,5]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[7,6]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[8,6]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[9,6]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    #For those hopping not listed in SK table
    #Compute the opposite hopping and use Hermiticity of Hamiltonian to 
    #derive these elements. 
    #Now flip the direction:
    l,m,n=nij.*-1
    #dd interaction
    V[3,2]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[4,2]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[5,2]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[6,2]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[4,3]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[5,3]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[6,3]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[5,4]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[6,4]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[6,5]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #p-p interaction
    V[8,7]=(l*m*pp_sig-l*m*pp_pi)
    V[9,7]=(l*n*pp_sig-l*n*pp_pi)
    V[9,8]=(m*n*pp_sig-m*n*pp_pi)
    #ds interaction
    V[2,1]=3^.5 * l*m*sd_sig
    V[3,1]=3^.5*l*n*sd_sig
    V[4,1]=3^.5*n*m*sd_sig  
    V[5,1]=3^.5/2*(l^2-m^2)*sd_sig
    V[6,1]=(n^2-.5*(l^2+m^2))*sd_sig
    #ps interaction
    V[7,1]=(l*sp_sig)
    V[8,1]=(m*sp_sig)
    V[9,1]=(n*sp_sig)
    #dp interaction
    V[2,7]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[2,8]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[2,9]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[3,7]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[3,8]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[3,9]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[4,7]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[4,8]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[4,9]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[5,7]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[5,8]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[5,9]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[6,7]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[6,8]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[6,9]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    return V
end 



function AuAg_getIntraHopping(Rij)
    if norm(Rij)<1e-6 
        return Diagonal([-10.5943, -11.0105, -11.0105, -11.0105, -11.0105, -11.0105, 2.3102, 2.3102, 2.3102]./2)
    end
    nij=Rij./norm(Rij)
    dij=norm(Rij)

    
    
    function getHoppingParameters(d)
        d0=(2.8890+2.8837)/2
        I0_ss_sig=(0.2888+0.5000)/2
        p_ss_sig=(3.3967+3.4130)/2
        ss_sig=I0_ss_sig*exp(-p_ss_sig*(d/d0-1))
        I0_sp_sig=(27.5677+47.4232)/2
        p_sp_sig=(99.1027+75.4943)/2
        sp_sig=I0_sp_sig*exp(-p_sp_sig*(d/d0-1))
        I0_sd_sig=(-0.5037+0.0339)/2
        p_sd_sig=(0.4269+1.8747)/2
        sd_sig=I0_sd_sig*exp(-p_sd_sig*(d/d0-1))
        I0_pp_sig=(0.3040+1.0660)/2
        p_pp_sig=(3.1565+4.5685)/2
        pp_sig=I0_pp_sig*exp(-p_pp_sig*(d/d0-1))
        I0_pp_pi=(0.2953+0.1271)/2
        p_pp_pi=(5.1796+6.1124)/2
        pp_pi=I0_pp_pi*exp(-p_pp_pi*(d/d0-1))
        I0_pd_sig=(99.9958+99.8976)/2
        p_pd_sig=(99.8750+99.9935)/2
        pd_sig=I0_pd_sig*exp(-p_pd_sig*(d/d0-1))
        I0_pd_pi=(-99.1472+-98.6434)/2
        p_pd_pi=(99.1178+91.1682)/2
        pd_pi=I0_pd_pi*exp(-p_pd_pi*(d/d0-1))
        I0_dd_sig=(0.7252+0.9232)/2
        p_dd_sig=(2.3977+1.8897)/2
        dd_sig=I0_dd_sig*exp(-p_dd_sig*(d/d0-1))
        I0_dd_pi=(0.2511+0.3693)/2
        p_dd_pi=(2.1463+2.0762)/2
        dd_pi=I0_dd_pi*exp(-p_dd_pi*(d/d0-1))
        I0_dd_del=(-0.9280+-0.9722)/2
        p_dd_del=(1.7029+1.3285)/2
        dd_del=I0_dd_del*exp(-p_dd_del*(d/d0-1))
        return sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del
        #return 0,0,0,0,0,0,0,0,0,0
    end
    l,m,n=nij
    sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del=getHoppingParameters(dij)
    V=zeros(ComplexF64,9,9)
    #Same orbital interaction
    V[1,1]=ss_sig
    V[2,2]=( 3*l^2*m^2*dd_sig + ( l^2 + m^2 - 4*l^2*m^2)*dd_pi +(n^2+l^2*m^2)*dd_del)
    V[3,3]=( 3*l^2*n^2*dd_sig + ( l^2 + n^2 - 4*l^2*n^2)*dd_pi +(m^2+l^2*n^2)*dd_del)
    V[4,4]=( 3*m^2*n^2*dd_sig + ( m^2 + n^2 - 4*m^2*n^2)*dd_pi +(l^2+m^2*n^2)*dd_del)
    V[5,5]=(3/4*(l^2-m^2)^2*dd_sig+(l^2+m^2-(l^2-m^2)^2)*dd_pi+(n^2+(l^2-m^2)^2/4)*dd_del)
    V[6,6]=((n^2-.5*(l^2+m^2))^2*dd_sig+3*n^2*(l^2+m^2)*dd_pi +3/4*(l^2+m^2)^2*dd_del)
    V[7,7]=(l^2*pp_sig+(1-l^2)*pp_pi)
    V[8,8]=(m^2*pp_sig+(1-m^2)*pp_pi)
    V[9,9]=(n^2*pp_sig+(1-n^2)*pp_pi)
    #p-p interaction
    V[7,8]=(l*m*pp_sig-l*m*pp_pi)
    V[7,9]=(l*n*pp_sig-l*n*pp_pi)
    V[8,9]=(m*n*pp_sig-m*n*pp_pi)
    #dd interaction
    V[2,3]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[2,4]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[2,5]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[2,6]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[3,4]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[3,5]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[3,6]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[4,5]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[4,6]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[5,6]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #s-d interaction
    V[1,2]=3^.5 * l*m*sd_sig
    V[1,3]=3^.5*l*n*sd_sig
    V[1,4]=3^.5*n*m*sd_sig
    V[1,5]=3^.5/2*(l^2-m^2)*sd_sig
    V[1,6]=(n^2-.5*(l^2+m^2))*sd_sig
    #sp interaction
    V[1,7]=(l*sp_sig)
    V[1,8]=(m*sp_sig)
    V[1,9]=(n*sp_sig)
    #pd interaction
    V[7,2]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[8,2]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[9,2]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[7,3]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[8,3]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[9,3]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[7,4]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[8,4]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[9,4]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[7,5]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[8,5]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[9,5]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[7,6]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[8,6]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[9,6]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    #For those hopping not listed in SK table
    #Compute the opposite hopping and use Hermiticity of Hamiltonian to 
    #derive these elements. 
    #Now flip the direction:
    #l,m,n=nij.*-1
    #dd interaction
    V[3,2]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[4,2]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[5,2]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[6,2]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[4,3]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[5,3]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[6,3]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[5,4]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[6,4]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[6,5]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #p-p interaction
    V[8,7]=(l*m*pp_sig-l*m*pp_pi)
    V[9,7]=(l*n*pp_sig-l*n*pp_pi)
    V[9,8]=(m*n*pp_sig-m*n*pp_pi)
    #ds interaction
    V[2,1]=3^.5 * l*m*sd_sig
    V[3,1]=3^.5*l*n*sd_sig
    V[4,1]=3^.5*n*m*sd_sig  
    V[5,1]=3^.5/2*(l^2-m^2)*sd_sig
    V[6,1]=(n^2-.5*(l^2+m^2))*sd_sig
    #ps interaction
    V[7,1]=(l*sp_sig)
    V[8,1]=(m*sp_sig)
    V[9,1]=(n*sp_sig)
    #dp interaction
    V[2,7]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[2,8]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[2,9]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[3,7]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[3,8]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[3,9]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[4,7]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[4,8]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[4,9]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[5,7]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[5,8]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[5,9]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[6,7]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[6,8]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[6,9]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    return Diagonal(V)
    #return V

end 


function AuAg_getInterHopping(Rij)
    nij=Rij./norm(Rij)
    dij=norm(Rij)
    function getHoppingParameters(d)
        d0=(2.8890+2.8837)/2
        V0_ss_sig=(-0.8864+-0.9261)/2
        q_ss_sig=(2.5004+3.4147)/2
        ss_sig=V0_ss_sig*exp(-q_ss_sig*(d/d0-1))
        V0_sp_sig=(1.2238+1.3669)/2
        q_sp_sig=(1.7035+3.0152)/2
        sp_sig=V0_sp_sig*exp(-q_sp_sig*(d/d0-1))
        V0_sd_sig=(-0.5268+-0.6941)/2
        q_sd_sig=(3.8742+4.0517)/2
        sd_sig=V0_sd_sig*exp(-q_sd_sig*(d/d0-1))
        V0_pp_sig=(1.5428+1.7926)/2
        q_pp_sig=(1.4366+2.5772)/2
        pp_sig=V0_pp_sig*exp(-q_pp_sig*(d/d0-1))
        V0_pp_pi=(-0.5098+-0.5155)/2
        q_pp_pi=(4.0872+2.9059)/2
        pp_pi=V0_pp_pi*exp(-q_pp_pi*(d/d0-1))
        V0_pd_sig=(-0.6058+-0.9479)/2
        q_pd_sig=(5.3603+4.2849)/2
        pd_sig=V0_pd_sig*exp(-q_pd_sig*(d/d0-1))
        V0_pd_pi=(0.1868+0.2972)/2
        q_pd_pi=(6.7768+4.6552)/2
        pd_pi=V0_pd_pi*exp(-q_pd_pi*(d/d0-1))
        V0_dd_sig=(-0.4540+-0.6844)/2
        q_dd_sig=(5.5990 +5.4403)/2
        dd_sig=V0_dd_sig*exp(-q_dd_sig*(d/d0-1))
        V0_dd_pi=(0.2456+0.3381)/2
        q_dd_pi=(5.2114 +5.0338)/2
        dd_pi=V0_dd_pi*exp(-q_dd_pi*(d/d0-1))
        V0_dd_del=(-0.0496+-0.0592)/2
        q_dd_del=(3.9194+2.4849)/2
        dd_del=V0_dd_del*exp(-q_dd_del*(d/d0-1))
        return sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del
    end 
    l,m,n=nij
    sp_sig,ss_sig,pp_sig,pp_pi,sd_sig,pd_sig,pd_pi,dd_sig,dd_pi,dd_del=getHoppingParameters(dij)
    V=zeros(ComplexF64,9,9)
    #Same orbital interaction
    V[1,1]=ss_sig
    V[2,2]=( 3*l^2*m^2*dd_sig + ( l^2 + m^2 - 4*l^2*m^2)*dd_pi +(n^2+l^2*m^2)*dd_del)
    V[3,3]=( 3*l^2*n^2*dd_sig + ( l^2 + n^2 - 4*l^2*n^2)*dd_pi +(m^2+l^2*n^2)*dd_del)
    V[4,4]=( 3*m^2*n^2*dd_sig + ( m^2 + n^2 - 4*m^2*n^2)*dd_pi +(l^2+m^2*n^2)*dd_del)
    V[5,5]=(3/4*(l^2-m^2)^2*dd_sig+(l^2+m^2-(l^2-m^2)^2)*dd_pi+(n^2+(l^2-m^2)^2/4)*dd_del)
    V[6,6]=((n^2-.5*(l^2+m^2))^2*dd_sig+3*n^2*(l^2+m^2)*dd_pi +3/4*(l^2+m^2)^2*dd_del)
    V[7,7]=(l^2*pp_sig+(1-l^2)*pp_pi)
    V[8,8]=(m^2*pp_sig+(1-m^2)*pp_pi)
    V[9,9]=(n^2*pp_sig+(1-n^2)*pp_pi)
    #p-p interaction
    V[7,8]=(l*m*pp_sig-l*m*pp_pi)
    V[7,9]=(l*n*pp_sig-l*n*pp_pi)
    V[8,9]=(m*n*pp_sig-m*n*pp_pi)
    #dd interaction
    V[2,3]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[2,4]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[2,5]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[2,6]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[3,4]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[3,5]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[3,6]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[4,5]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[4,6]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[5,6]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #s-d interaction
    V[1,2]=3^.5 * l*m*sd_sig
    V[1,3]=3^.5*l*n*sd_sig
    V[1,4]=3^.5*n*m*sd_sig
    V[1,5]=3^.5/2*(l^2-m^2)*sd_sig
    V[1,6]=(n^2-.5*(l^2+m^2))*sd_sig
    #sp interaction
    V[1,7]=(l*sp_sig)
    V[1,8]=(m*sp_sig)
    V[1,9]=(n*sp_sig)
    #pd interaction
    V[7,2]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[8,2]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[9,2]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[7,3]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[8,3]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[9,3]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[7,4]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[8,4]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[9,4]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[7,5]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[8,5]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[9,5]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[7,6]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[8,6]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[9,6]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    #For those hopping not listed in SK table
    #Compute the opposite hopping and use Hermiticity of Hamiltonian to 
    #derive these elements. 
    #Now flip the direction:
    l,m,n=nij.*-1
    #dd interaction
    V[3,2]=(3*l^2*m*n*dd_sig+m*n*(1-4*l^2)*dd_pi+m*n*(l^2-1)*dd_del) #xy-xz
    V[4,2]=(3*l*m^2*n*dd_sig+l*n*(1-4*m^2)*dd_pi+l*n*(m^2-1)*dd_del) #xy-yz
    V[5,2]=(1.5*l*m*(l^2-m^2)*dd_sig+2*l*m*(m^2-l^2)*dd_pi+.5*l*m*(l^2-m^2)*dd_del) #xy - x2-y2
    V[6,2]=(3^.5*l*m*(n^2-.5*(l^2+m^2))*dd_sig-2*3^.5*l*m*n^2*dd_pi+.5*3^.5*l*m*(1+n^2)*dd_del) #xy - 3z^2-r^2
    V[4,3]=(3*n^2*m*l*dd_sig+m*l*(1-4*n^2)*dd_pi+l*m*(n^2-1)*dd_del) #xz-yz
    V[5,3]=(1.5*n*l*(l^2-m^2)*dd_sig+n*l*(1-2*(l^2-m^2))*dd_pi-n*l*(1-.5*(l^2-m^2))*dd_del) #xz ->x^2-y^2
    V[6,3]=(3^.5*l*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*l*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*l*n*(l^2+m^2)*dd_del) #xz ->z^2
    V[5,4]=(1.5*m*n*(l^2-m^2)*dd_sig-m*n*(1+2*(l^2-m^2))*dd_pi+m*n*(1+(l^2-m^2)/2)*dd_del) #yz ->x^2-y^2
    V[6,4]=(3^.5*m*n*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*m*n*(l^2+m^2-n^2)*dd_pi-.5*3^.5*m*n*(l^2+m^2)*dd_del)#yz->z^2
    V[6,5]=(.5*3^.5*(l^2-m^2)*(n^2-.5*(l^2+m^2))*dd_sig+3^.5*n^2*(m^2-l^2)*dd_pi+3^.5*(1+n^2)*(l^2-m^2)/4*dd_del)
    #p-p interaction
    V[8,7]=(l*m*pp_sig-l*m*pp_pi)
    V[9,7]=(l*n*pp_sig-l*n*pp_pi)
    V[9,8]=(m*n*pp_sig-m*n*pp_pi)
    #ds interaction
    V[2,1]=3^.5 * l*m*sd_sig
    V[3,1]=3^.5*l*n*sd_sig
    V[4,1]=3^.5*n*m*sd_sig  
    V[5,1]=3^.5/2*(l^2-m^2)*sd_sig
    V[6,1]=(n^2-.5*(l^2+m^2))*sd_sig
    #ps interaction
    V[7,1]=(l*sp_sig)
    V[8,1]=(m*sp_sig)
    V[9,1]=(n*sp_sig)
    #dp interaction
    V[2,7]=(3^.5*l^2*m*pd_sig+m*(1-2*l^2)*pd_pi)#x->xy 
    V[2,8]=(3^0.5*m^2*l*pd_sig+l*(1-2*m^2)*pd_pi)#y->xy
    V[2,9]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi)#z->xy
    V[3,7]=(3^.5*l^2*n*pd_sig+n*(1-2*l^2)*pd_pi) #x->xz
    V[3,8]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #xz->y
    V[3,9]=(3^0.5*n^2*l*pd_sig+l*(1-2*n^2)*pd_pi) #xz->z
    V[4,7]=(3^0.5*l*m*n*pd_sig-2*l*m*n*pd_pi) #yz->x
    V[4,8]=(3^0.5*m^2*n*pd_sig+n*(1-2*m^2)*pd_pi) #yz->y
    V[4,9]=(3^0.5*n^2*m*pd_sig+m*(1-2*n^2)*pd_pi) #yz->z
    V[5,7]=(3^0.5/2*l*(l^2-m^2)*pd_sig+l*(1-l^2+m^2)*pd_pi) #x^2-y^2->x
    V[5,8]=(3^0.5/2*m*(l^2-m^2)*pd_sig-m*(1+l^2-m^2)*pd_pi) #x^2-y^2->y
    V[5,9]=(3^0.5/2*n*(l^2-m^2)*pd_sig-n*(l^2-m^2)*pd_pi) #x^2-y^2->z
    V[6,7]=(l*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*l*n^2*pd_pi) #3z^2-r^2->x
    V[6,8]=(m*(n^2-(l^2+m^2)/2)*pd_sig-3^0.5*m*n^2*pd_pi) #3z^2-r^2->y
    V[6,9]=(n*(n^2-(l^2+m^2)/2)*pd_sig+3^0.5*n*(l^2+m^2)*pd_pi)#3z^2-r^2->z
    return V
end 




function addV!(iidx,jidx,value,V,iptr,jptr)
    # Add the values of the V 9x9 matrix into a sparse matrix format
    # iidx (jidx) is the set of (lines,columns) of the sparse matrix
    # iptr, jptr index the atoms
    local i = 0
    local j = 0
    local vv = 0.0
    local a,b = size(V)
    for i = 1:a
        for j = 1:b
            vv = V[i,j]
            if abs(vv) > 1e-10
                push!(iidx, iptr+i)
                push!(jidx, jptr+j)
                push!(value, vv)
            end 
        end 
    end 
end


function construct_alloy_hr(nntable_idx,nntable_dir,functionDict,typeslist)
    Hr_iidx=Array{Int64,1}([])
    Hr_jidx=Array{Int64,1}([])
    Hr_value=Array{ComplexF64,1}([])
    nn_relative_pos=[[1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],[0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1]]*a_Au/2
    for i=1:lastindex(nntable_idx)
        let 
            Atomi=typeslist[i]
            Vi=functionDict[["intra",Atomi,Atomi]]([0.0;0.0;0.0])
            
            if length(nntable_idx[i])<12  # checking if it is a surface atom, or else it does the normal implementation
                
                idx_list = []
                for k=1:lastindex(nntable_dir[i])
                    for l=1:12
                       if isapprox(nntable_dir[i][k],nn_relative_pos[l])
                           push!(idx_list,l) # list of neighbours that are "real" 
                        end
                    end
                end
                
                # first we calculate the hamiltonian from the existing neighbours - the normal implementation of the inter and intra atomic energies for the real neighbours
                for j_aux=1:lastindex(nntable_idx[i])
                    j=nntable_idx[i][j_aux]
                    iptr=(i-1)*9 #9 is the number of orbitals in Au, Ag and Cu 
                    jptr=(j-1)*9
                    Atomj=typeslist[j]
                    V=functionDict[["inter",Atomi,Atomj]](nntable_dir[i][j_aux])
                    addV!(Hr_iidx,Hr_jidx,Hr_value,V,iptr,jptr)
                    Vi=Vi+functionDict[["intra",Atomi,Atomj]](nntable_dir[i][j_aux])
                end
                
                # now we add the the effect of the virtual neighbours to the intra atomic energy of the site.
                for index=1:12
                    if index ∉ idx_list # checking if the neighbour is real or virtual
                        num=rand() # keeping the virtual neighbour composition same as the rest of the alloy
                        if num<Au_frac 
                            neighbour = "Au"
                        else 
                            neighbour = "Ag"
                        end
                        Vi=Vi+functionDict[["intra",Atomi,neighbour]](nn_relative_pos[index]) # adding the contributions of each virtual neighbour to the intra atomic energies
                    end
                end
            else
                for j_aux=1:lastindex(nntable_idx[i])
                    j=nntable_idx[i][j_aux]
                    iptr=(i-1)*9 #9 is the number of orbitals in Au, Ag and Cu 
                    jptr=(j-1)*9
                    Atomj=typeslist[j]
                    V=functionDict[["inter",Atomi,Atomj]](nntable_dir[i][j_aux])
                    addV!(Hr_iidx,Hr_jidx,Hr_value,V,iptr,jptr)
                    Vi=Vi+functionDict[["intra",Atomi,Atomj]](nntable_dir[i][j_aux])
                end
            end
            addV!(Hr_iidx,Hr_jidx,Hr_value,Vi,(i-1)*9,(i-1)*9)
        end
    end 

    return Hr_iidx,Hr_jidx,Hr_value 
end

function construct_alloy_poslist(pos_list,Rcut;lattice=[[1.0;0;0] [0;1.0;0] [0;0;1.0]])
    #=
    pos_list(3,N), A list that stores the CARTESIAN coordinate of each atoms. where N is the number of positions in your atom
    typeslist(N), the type of each atom 
    functiondict, A dictionary that gives you the function to use given the hopping function given the alloy type 
    =#
    let 
        pos_list_true=lattice*pos_list
        neighbor_idx_list=[]
        neighbor_dir_list=[]
        for i=1:size(pos_list)[2]
            push!(neighbor_dir_list,[])
            push!(neighbor_idx_list,[])
        end
        postree=BallTree(pos_list_true)#Use KDtree to find the neighbors 
        for i=1:size(pos_list_true)[2]
            point=pos_list_true[:,i]
            neighbors=inrange(postree,point,Rcut)
            for neighbor in neighbors 
                push!(neighbor_idx_list[i],neighbor)
                push!(neighbor_dir_list[i],pos_list_true[:,neighbor]-point)
            end 
        end
        
        for i=1:lastindex(neighbor_idx_list)
            del_index=nothing
            for j=1:lastindex(neighbor_idx_list[i])
                if i==neighbor_idx_list[i][j]
                   del_index=j
                end
            end
            deleteat!(neighbor_idx_list[i],del_index)
            deleteat!(neighbor_dir_list[i],del_index)
        end

        return neighbor_idx_list,neighbor_dir_list
    end


end


function get_alloy_diel(GMF,omega)
    # DielectricModel computes the dielectric function of a Gold-Silver alloy
    # with gold molar fraction equal to GMF; according to the presented model.
    # GMF takes values between 0 [pure silver] & 1 [pure gold].
    # Example: [Lambda, AuAg5050] = DielectricModel[0.5]
    c=2.99792458e17; # Speed of light; in nm/s
    h=4.135667516e-15; # Planck's constant; in eV.s
    lambda=270:1:650; # In nm
    #omega=h*c./lambda; # In eV
    ModelParameters=[8.9234 8.5546 9.0218; # wp
     0.042389 0.022427 0.16713; # gammap
     2.2715 1.7381 2.2838; # einf
     2.6652 4.0575 3.0209; # wg1
     2.3957 3.9260 2.7976; # w01
     0.17880 0.017723 0.18833; # gamma1
     73.251 51.217 22.996; # A1
     3.5362 4.1655 3.3400; # w02
     0.35467 0.18819 0.68309; # gamma2
     40.007 30.770 57.540]  # A2
    wp = GMF^2*(2*ModelParameters[1,1]-4*ModelParameters[1,3]+2*ModelParameters[1,2])+  GMF*(-ModelParameters[1,1]+4*ModelParameters[1,3]-3*ModelParameters[1,2]) +  ModelParameters[1,2]
    gammap = GMF^2*(2*ModelParameters[2,1]-4*ModelParameters[2,3]+2*ModelParameters[2,2]) +  GMF*(-ModelParameters[2,1]+4*ModelParameters[2,3]-3*ModelParameters[2,2]) +  ModelParameters[2,2]
    einf = GMF^2*(2*ModelParameters[3,1]-4*ModelParameters[3,3]+2*ModelParameters[3,2])+  GMF*(-ModelParameters[3,1]+4*ModelParameters[3,3]-3*ModelParameters[3,2]) + ModelParameters[3,2]
    wg1 = GMF^2*(2*ModelParameters[4,1]-4*ModelParameters[4,3]+2*ModelParameters[4,2]) +  GMF*(-ModelParameters[4,1]+4*ModelParameters[4,3]-3*ModelParameters[4,2]) +  ModelParameters[4,2]
    w01 = GMF^2*(2*ModelParameters[5,1]-4*ModelParameters[5,3]+2*ModelParameters[5,2]) +  GMF*(-ModelParameters[5,1]+4*ModelParameters[5,3]-3*ModelParameters[5,2]) +  ModelParameters[5,2]
    gamma1 = GMF^2*(2*ModelParameters[6,1]-4*ModelParameters[6,3]+2*ModelParameters[6,2]) +  GMF*(-ModelParameters[6,1]+4*ModelParameters[6,3]-3*ModelParameters[6,2]) +  ModelParameters[6,2]
    A1 = GMF^2*(2*ModelParameters[7,1]-4*ModelParameters[7,3]+2*ModelParameters[7,2])+  GMF*(-ModelParameters[7,1]+4*ModelParameters[7,3]-3*ModelParameters[7,2]) +  ModelParameters[7,2]
    w02 = GMF^2*(2*ModelParameters[8,1]-4*ModelParameters[8,3]+2*ModelParameters[8,2]) +  GMF*(-ModelParameters[8,1]+4*ModelParameters[8,3]-3*ModelParameters[8,2]) + ModelParameters[8,2]
    gamma2 = GMF^2*(2*ModelParameters[9,1]-4*ModelParameters[9,3]+2*ModelParameters[9,2]) +  GMF*(-ModelParameters[9,1]+4*ModelParameters[9,3]-3*ModelParameters[9,2]) +  ModelParameters[9,2]
    A2 = GMF^2*(2*ModelParameters[10,1]-4*ModelParameters[10,3]+2*ModelParameters[10,2]) + GMF*(-ModelParameters[10,1]+4*ModelParameters[10,3]-3*ModelParameters[10,2]) +  ModelParameters[10,2]
    Drude = einf .- ((wp^2)./((omega.^2) .+ 1im*gammap*omega))
    CP1 = A1*((1 ./((omega.+1im*gamma1).^2)) .* (-sqrt.(omega.+1im*gamma1.-wg1).*atan.(sqrt.((wg1-w01)./(omega.+1im*gamma1.-wg1))))  + (1 ./((omega.+1im*gamma1).^2)) .* (-sqrt.(omega.+1im*gamma1.+wg1).*atanh.(sqrt.((wg1-w01)./(omega.+1im*gamma1.+wg1))))  + (1 ./((omega.+1im*gamma1).^2)) .* (2*sqrt(wg1)*atanh(sqrt((wg1-w01)/wg1)))  - sqrt(wg1-w01).*log.(1 .-((omega.+1im*gamma1)/w01).^2)./(2*(omega.+1im*gamma1).^2))
    CP2 = - A2 .* log.(1 .-((omega.+1im*gamma2)/w02).^2)./(2*(omega.+1im*gamma2).^2)
    DielectricFunction = Drude + CP1 + CP2
    #return lambda, DielectricFunction
    return DielectricFunction
end




function get_alloy_hamiltonian(radius, lattice_param, gold_fraction)
    #define the functionDict
    nnlist=[[1,1,0];;[-1,1,0];;[1,-1,0];;[-1,-1,0];;[1,0,1];;[-1,0,1];;[1,0,-1];;[-1,0,-1];;[0,1,1];;[0,-1,1];;[0,1,-1];;[0,-1,-1]]

    Rmax=radius # nanoparticle size in Angstrom
    global a_Au=lattice_param # lattice parameter of gold in Angstrom
    function in_shape(R,vararg)
        Rmax=vararg[1]
        a_Au=vararg[2]
        R_actual=R.*(a_Au/2)
        if norm(R_actual)<Rmax 
            return true 
        else 
            return false 
        end
    end
    
    Elist,Edict=explore_alloy(nnlist,in_shape,Rmax,a_Au)
    R = Deque_to_vec(Elist, Edict; Transform=Transform)*a_Au/2
    
    #Now convert Elist and to array 
    pos_list=zeros(3,length(Elist))
    for i=1:length(Elist)
        pos_list[:,i]=R[i,:]
    end
    global Au_frac=gold_fraction
    #Construct the typeslist
    typeslist=[]
    for i=1:length(Elist)
        let
            num=rand() 
            if num<Au_frac 
                push!(typeslist,"Au")
            else 
                push!(typeslist,"Ag")
            end
        end
    end

    global functionDict=Dict()


    global functionDict[["inter","Au","Au"]]=Au_getInterHopping
    global functionDict[["intra","Au","Au"]]=Au_getIntraHopping
    global functionDict[["inter","Ag","Ag"]]=Ag_getInterHopping
    global functionDict[["intra","Ag","Ag"]]=Ag_getIntraHopping
    global functionDict[["inter","Au","Ag"]]=AuAg_getInterHopping
    global functionDict[["intra","Au","Ag"]]=AuAg_getIntraHopping
    global functionDict[["inter","Ag","Au"]]=AuAg_getInterHopping
    global functionDict[["intra","Ag","Au"]]=AuAg_getIntraHopping
    
    
    neighbor_idx_list,neighbor_dir_list=construct_alloy_poslist(pos_list,a_Au*1.01/sqrt(2))
    Hr_iidx,Hr_jidx,Hr_value=construct_alloy_hr(neighbor_idx_list,neighbor_dir_list,functionDict,typeslist)
    
    Hr_value=Hr_value./27.211386245988 # converting from eV to Ha

    H=sparse(Hr_iidx,Hr_jidx,Hr_value)
   
    ll = -12.5
    ul = 11.5
    lambda = (ul+ll)/2/27.211386245988
    delta = (ul-ll)/27.211386245988
    lam_I = UniformScaling(lambda)
    
    H = (H - lam_I)./delta

    R = R./10 # converting from angstrom to nm

    return H, R

end
