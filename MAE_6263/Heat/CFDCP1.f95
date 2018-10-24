!Course Project I for MAE 6263
!Explicit and Implicit Solver for 2-D Heat Equation
!Implicit Solver uses Jacobi/Gauss - Seidel/Iterative Method
!ICs - u(x,y) = 0 everywhere except boundaries
!BCs - u(x,y) = x+y through time
!Romit Maulik, PhD Student, CFD Laboratory, OSU

program CFDCP1
implicit none

integer :: nspace,ns,i,j,k,isol,gsnt,msol,insol,fostep,outpar
real*8 :: ft,tl,alpha,dspace,dt,gm,tol,t1,t2
real*8,allocatable,dimension(:,:) :: u


common /diffusivity/ alpha
common/gstimestep/ gsnt
common/msolver/ msol
common/insolver/ insol
common/contol/ tol


open(15,file='CFDCP1Input.txt')
read(15,*)nspace	!number of spatial grid points x & y
read(15,*)tl		!Total Length 
read(15,*)gm		!Stability factor
read(15,*)ft		!Final Time
read(15,*)alpha		!Diffusivity
read(15,*)isol		!isol;	Solver - [0] - For explicit [1] - For Implicit
read(15,*)gsnt		!gsnt; 	Number of Iterative Steps in GS or Jacobi
read(15,*)fostep	!fostep;File output every these many timesteps
read(15,*)insol		!insol;	Type of Solver - Iversion [0] or Iterative [1]
read(15,*)msol		!msol; Matrix Solver [0] - jacobi [1] - Gauss Seidel
read(15,*)tol		!tol; Convergence tolerance
close(15)

!Calculating dx and dy
dspace = tl/dfloat(nspace)

!Calculating timestep
dt = gm*(dspace**2)/(alpha)

!Total number of timesteps
ns = nint(ft/dt)

!Stability Check
if (isol == 0) then
if (dt/(dspace**2)>0.25d0) then
  print*,'Unstable - Reduce Timestep'
  print*,dt
!  call exit(10)
end if
end if

!Initial Condition Setup
allocate(u(0:nspace,0:nspace))

do i = 1,nspace-1
  do j = 1,nspace-1
    u(i,j) = 0
  end do
end do

!Boundary Condition Setup

do i = 0,nspace
  u(i,0) = dfloat(i)/dfloat(nspace)
  u(i,nspace) = dfloat((i+nspace))/dfloat(nspace)
end do

do j = 0,nspace
  u(0,j) = dfloat(j)/dfloat(nspace)
  u(nspace,j) = dfloat((j+nspace))/dfloat(nspace)
end do

!IC File Ouput
open(20,file="InitialField.plt")
write(20,*)'Title="IC Data set"'
write(20,*)'variables ="x","y","u"'
close(20)

open(20,file="InitialField.plt",position="append")
write(20,"(a,i8,a,i8,a)")'Zone I = ',nspace+1,',J=',nspace+1,',F=POINT'
  do j = 0,nspace
    do i = 0,nspace
      write (20, '(1600F14.3)',advance="no")dfloat(i),dfloat(j),u(i,j)
      write(20,*) ''
    end do
  end do
close(20)



!Output file setup
open(20,file="ContourPlots.plt")
write(20,*)'Title="Transient data set"'
write(20,*)'variables ="x","y","u"'
close(20)

!Output file setup at y=0.5
open(20,file="LinePlots.plt")
write(20,*)'Title="Transient data set"'
write(20,*)'variables ="x","u"'
close(20)


call cpu_time(t1)
!Time integration - 
do k = 1,ns

if (isol == 0) then
	call heatexp(u,nspace,dspace,dt)	
else if (isol == 1) then
  	call heatimp(u,nspace,dspace,dt)
end if



outpar = ns*fostep/100

  
!Output to transient .plt file for Tecplot  
if (mod(k,outpar)==0) then
open(20,file="ContourPlots.plt",position="append")
write(20,"(a,i8,a,i8,a)")'Zone I = ',nspace+1,',J=',nspace+1,',F=POINT'
write(20,"(a,i8)")'StrandID=0,SolutionTime=',k
  do j = 0,nspace
    do i = 0,nspace
      write (20, '(1600F14.3,1600F14.3,1600F14.3)',advance="no")dfloat(i),dfloat(j),u(i,j)
      write(20,*) ''
    end do
  end do
close(20)

open(20,file="LinePlots.plt",position="append")
write(20,"(a,i8,a,i8,a)")'Zone I = ',nspace+1,',F=POINT'
write(20,"(a,i8)")'StrandID=0,SolutionTime=',k
    do i = 0,nspace
      write (20, '(1600F14.3,1600F14.3)',advance="no")dfloat(i),u(i,nspace/2)
      write(20,*) ''
    end do
close(20)

end if


 
end do

call cpu_time(t2)

open(4,file='cpu.txt')
write(4,*)"cpu time (sec)=",(t2-t1)
close(4)


end

!--------------------------------------------------------------------------------------
!End of Program
!--------------------------------------------------------------------------------------




!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for Explicit Solver
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine heatexp(u,nspace,dspace,dt)
implicit none

integer :: nspace,i,j
real*8 :: dt,dspace,alpha,ta,tb
real*8,dimension(0:nspace,0:nspace):: u,v

common/diffusivity/ alpha

!Explicit algebraic solving - point to point

do i = 0,nspace
  do j = 0,nspace
	v(i,j) = u(i,j)
  end do
end do

do i = 1,nspace-1
  do j = 1,nspace-1
    ta =  u(i+1,j)-2d0*u(i,j)+u(i-1,j)
    tb =  u(i,j+1)-2d0*u(i,j)+u(i,j-1)
	v(i,j) = alpha*(ta+tb)*dt/(dspace**2)+u(i,j)
  end do
end do

do j = 1,nspace-1
  do i = 1,nspace-1
    u(i,j) = v(i,j)
  end do
end do

return
end


!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for Implicit Solver - Matrix preparation for Inversion
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine heatimp(u,nspace,dspace,dt)
implicit none


common/diffusivity/ alpha
common/msolver/ msol
common/insolver/ insol

integer :: nspace,i,j,cnt,k,l,msol,insol!,ad
real*8 :: dt,dspace,alpha
real*8,dimension(0:nspace,0:nspace)::u
real*8,allocatable,dimension(:)::urhs,v!,vnew
real*8,allocatable,dimension(:)::md,fodl,fodr,sodl,sodr


if (insol == 0) then

l = (nspace-1)**2-1

allocate(urhs(0:l))
allocate(v(0:l))
allocate(md(0:l))
allocate(fodl(0:l))
allocate(fodr(0:l))
allocate(sodl(0:l))
allocate(sodr(0:l))

!Building RHS Vector - Consult node numbering from Moin textbook
!Lower Boundary
urhs(0) = u(1,1)+(alpha*dt/dspace**2)*u(0,1)+(alpha*dt/dspace**2)*u(1,0)

do i = 1,nspace-3
  urhs(i) = u(i+1,1) + (alpha*dt/dspace**2)*u(i+1,0)
end do

urhs(nspace-2) = u(nspace-1,1)+(alpha*dt/dspace**2)*u(nspace-1,0)+(alpha*dt/dspace**2)*u(nspace,1)


!Middle Region
cnt = nspace-1

do j = 2,nspace-2
	urhs(cnt) = u(1,j) + (alpha*dt/dspace**2)*u(0,j)
		do i = 2,nspace-2
			urhs(cnt+i-1) = u(i,j)
        end do
    urhs(cnt+nspace-2) = u(nspace-1,j)+(alpha*dt/dspace**2)*u(nspace,j)
	cnt = cnt + (nspace-1)
end do

!Upper Boundary

urhs(cnt) = u(1,nspace-1)+(alpha*dt/dspace**2)*u(0,nspace-1)+(alpha*dt/dspace**2)*u(1,nspace)

do i = 1,nspace-3
  urhs(cnt+i) = u(i+1,nspace-1)+(alpha*dt/dspace**2)*u(i+1,nspace)
end do

urhs(cnt+nspace-2) = u(nspace-1,nspace-1)+(alpha*dt/dspace**2)*u(nspace-1,nspace)+(alpha*dt/dspace**2)*u(nspace,nspace-1)


!Check URHS
!open(20,file="URHS.txt")
!write(20,'(A)')'URHS Check'
!write(20,*)''
!close(20)


!do i = 0,(nspace-1)**2-1
!  open(20,file="URHS.txt",position="append",status="replace")
!  write(20,'(1600F14.3)')(urhs(i))
!  close(20)
!end do

!call exit(1)

!Now URHS completely defined


!Defining Block Tridiagonal Matrix Next

!Main diagonal of matrix

do i = 0,l
	md(i) = 1d0 + 4.*(alpha*dt/dspace**2)
end do

!First off diagonal left side
do i = l,0,-1
  fodl(i) = -(alpha*dt/dspace**2)
end do

do i = 0,l,(nspace-1)
  fodl(i) = 0d0
end do

!First off diagonal right side
do i = l,0,-1
fodr(i) = -(alpha*dt/dspace**2)
end do

do i = l,(nspace-2),-(nspace-1)
  fodr(i) = 0
end do

!Second off diagonal left side
do i = l,0,-1
  sodl = -(alpha*dt/dspace**2)
end do

do i = 0,(nspace-2)
sodl(i) = 0d0
end do

!Second off diagonal right side
do i = l,0,-1
  sodr = -(alpha*dt/dspace**2)
end do

do i = l,l-(nspace-2),-1
sodr(i) = 0d0
end do

!Check A matrix
!open(20,file="MD.txt")
!write(20,'(A)')'MD Check'
!write(20,*)''
!close(20)


!do i = 0,l
!  open(20,file="MD.txt",position="append",status="replace")
!  write(20,'(1600F14.3)')(sodl(i))
!  close(20)
!end do

!call exit(1)


!Matrix Inverse Iterations Next

k = 0
do j = 1,nspace-1
  do i = 1,nspace-1
    v(k) = u(i,j)
    k = k + 1
  end do
end do  

if (msol == 0) then
	call jacobi(md,sodl,sodr,fodl,fodr,urhs,v,nspace,l)
else if (msol == 1) then
	call gseidel(md,sodl,sodr,fodl,fodr,urhs,v,nspace,l)
end if

deallocate(fodl,fodr,sodl,sodr,urhs)

k = 0

do j = 1,nspace-1
  do i = 1,nspace-1
    u(i,j) = v(k)
    k = k + 1
  end do
end do


else if (insol==1) then

call jciterative(u,nspace,dspace,dt)

end if
     
return
end



!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for Jacobi Solver
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine jacobi(md,sodl,sodr,fodl,fodr,urhs,v,nspace,l)
implicit none

common/gstimestep/ gsnt

integer::nspace,l,k,gsnt,j,i,nel,q
real*8,dimension(0:l)::urhs,v1,v
real*8,dimension(0:l)::md,fodl,sodl,fodr,sodr
real*8::gsum1

do i = 0,l
  v1(i) = v(i)
end do

do k = 0,gsnt

do i = 0,l

gsum1 = 0.

do j = 0,l
  if (j==i-1) then
    gsum1 = gsum1+v(j)*fodr(j)
  else if (j==i+1) then
    gsum1 = gsum1+v(j)*fodl(j)
  else if (j==i-(nspace-1)) then
    gsum1 = gsum1+v(j)*sodr(j)
  else if (j==i+(nspace-1)) then
    gsum1 = gsum1+v(j)*sodl(j)    
  end if
end do

v1(i) = 1./(md(i))*(urhs(i)-gsum1)
end do


!L1Norm check
nel = (nspace-1)**2-1
q = 0
call l1check(v1,v,nel,q)

if (q.ne.0) then
  exit
end if

do i = 0,(nspace-1)**2-1
  v(i) = v1(i)
end do


end do  



return
end


!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for Gauss-Seidel Solver
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine gseidel(md,sodl,sodr,fodl,fodr,urhs,v,nspace,l)
implicit none

common/gstimestep/ gsnt

integer::nspace,l,k,gsnt,j,i,q
real*8,dimension(0:l)::urhs,v,vnew
real*8,dimension(0:l)::md,fodl,fodr,sodl,sodr
real*8::gsum1,gsum2

do i = 0,l
  vnew(i) = 0d0
end do  

do k = 0,gsnt

do i = 0,(nspace-1)**2-1

gsum1 = 0.
gsum2 = 0.

do j = 0,(nspace-1)**2-1
	if (j<i) then
		if (j==i-1) then
    		gsum1 = gsum1+vnew(j)*fodr(j)
  		else if (j==i-(nspace-1)) then
    		gsum1 = gsum1+vnew(j)*sodr(j)
  		end if
	else if (j>i) then
    	if (j==i+1) then
    		gsum2 = gsum2+v(j)*fodl(j)
  		else if (j==i+(nspace-1)) then
    		gsum2 = gsum2+v(j)*sodl(j)    
  		end if        				
    end if
end do

vnew(i) = 1./(md(i))*(urhs(i)-gsum1-gsum2)
end do

!L1Norm check
q = 0
call l1check(vnew,v,l,q)

if (q.ne.0) then
  exit
end if

do i = 0,(nspace-1)**2
  v(i) = vnew(i)
end do

end do  



return
end


!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for Jacobi Iterative Solver
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine jciterative(u,nspace,dspace,dt)
implicit none

common/gstimestep/ gsnt
common/diffusivity/ alpha
common/contol/ tol

integer::nspace,k,gsnt,i,j
real*8::dspace,dt,beta,alpha,beta2,sum,tol
real*8,dimension(0:nspace,0:nspace)::u,v,v1


beta = alpha/(dspace**2)*dt

beta2 = beta/(1+4.*beta)

do i = 0,nspace
  do j = 0,nspace
    v(i,j) = u(i,j)
    v1(i,j) = u(i,j)
  end do
end do

do k = 1,gsnt

  do i = 1,nspace-1
    do j = 1,nspace-1
      v(i,j) = v1(i,j)
    end do
  end do
   
  do i = 1,nspace-1
    do j = 1,nspace-1
      v1(i,j) = beta2*(u(i+1,j)+u(i-1,j)+u(i,j+1)+u(i,j-1))+1/(1+4.*beta)*(u(i,j))
    end do
  end do

sum = 0.
  do i = 1,nspace-1
    do j = 1,nspace-1
		sum = sum + (v1(i,j)-v(i,j))
    end do
  end do

if (sum<tol) then
  exit
end if

end do

do i = 0,nspace
  do j = 0,nspace
    u(i,j) = v(i,j)
  end do
end do


return
end


!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
!Subroutine for L1 Norm Check
!--------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------
subroutine l1check(v1,v,nel,q)
implicit none

common/contol/ tol

integer::q,i,nel
real*8,dimension(0:nel)::v1,v
real*8::sum,tol

sum = 0.
do i = 0,nel
  sum = sum + (v1(i)-v(i))
end do

if (sum<tol) then
  q = 1
end if

return
end

