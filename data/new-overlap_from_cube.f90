!   Bai Shuming 2019.6.4 Duke University, Durham, NC, USA
!   Calculating the MO overlap from the cube file. The cube file has been transformed to (x,y,x,density) formet by cube2xyz program. 
!   Designed to calculate the overlap of HOMO-HOMO, LUMO-LUMO, and HOMO-HOMO*LUMO-LUMO. 

module params
implicit none
type den
 real*8  x,y,z,density
end type den
! type(den) trans(1000000)
end module params
 
program overlap_cube
use params
implicit none
integer i,j,k,nx,ny,nz,N,Ni,Npoint,Nhomo,Nlumo, natom, ncount
character*79:: record
real*8 stepx,stepy,stepz,tol_density,trans_origin(3),rot_dipole(3),dv1,transition(3),rotation1(3,3),transition2(3),rotation2(3,3)
real*8,allocatable :: overlaph(:), overlapl(:), overlaphl(:)
integer x1,y1,z1,x2,y2,z2,n1,n2
real*8 density1,density2, couple, distance, couple0,couple1
real*8,parameter::au2cm=219474.63068d0
real*8,parameter::au2angstrom=0.529177249
real*8 tol_x,tol_y,tauol_zi, xyz_origin(3), thresh
integer, allocatable :: if_takenl(:),if_takenh(:)
character*20 nonsense
real*8,allocatable :: homo(:)
real*8,allocatable :: homo2(:)
real*8,allocatable :: lumo(:)
real*8,allocatable :: lumo2(:)

open(unit=111,file='input.dat')
read(111,*) Npoint.
read(111,*) trans_origin(1), trans_origin(2), trans_origin(3)
read(111,*) transition(1), transition(2), transition(3)
read(111,*) thresh

trans_origin(:) = trans_origin(:)/au2angstrom
transition(:) = transition(:)/au2angstrom

    allocate(overlapl(Npoint))
    allocate(overlaph(Npoint))
    allocate(overlaphl(Npoint))

   open(unit=20,file='homo-xyz.dat')
       read(20,*) nonsense
       read(20,*) natom, xyz_origin(:)
       read(20,*) nx, stepx
       read(20,*) ny, stepy
       read(20,*) nz, stepz
       
       dv1 = stepx*stepy*stepz
       write(*,*) "dv1 = ", dv1

     open(unit=30,file='lumo-xyz.dat')
       read(30,*) nonsense
       read(30,*) natom, xyz_origin(:)
       read(30,*) nx, stepx
       read(30,*) ny, stepy
       read(30,*) nz, stepz  

 
       if(abs(stepx*stepy*stepz-dv1) .gt. 0.01) then
          write(*,*) "Wrong: HOMO and LUMO have different grid size!!!"
          stop
       end if
      
    N=nx*ny*nz

    allocate(homo(N))
    allocate(homo2(N))
    allocate(lumo(N))
    allocate(lumo2(N))
    allocate(if_takenl(N))
    allocate(if_takenh(N))    

    do i = 1, N
       read(20,*) homo(i)
       read(30,*) lumo(i)
    end do 
!$OMP END parallel DO
    close(111)
    close(20)
    close(30)

!Calculate the overlap 
       overlapl = 0.d0
       overlaph = 0.d0
       overlaphl = 0.d0

 open(444,file="overlap_calculated.dat")
 write(444,*) "distance overlap_homo overlap_lumo overlap_homo*overlap_lumo"

 open(555,file="abs_overlap_calculated.dat")
 write(555,*) "distance |overlap_homo| |overlap_lumo| |overlap_homo*overlap_lumo|"
 
 do Ni =  1, Npoint
!      get the two cube fixed in geometry
!      calculate the overlp by the xyz position.    
     ncount = 0
      do i = 1, nx
        do j = 1, ny
          do k = 1, nz
            n1 = (i-1)*ny*nz+(j-1)*nz+k
            if(abs(homo(n1)) .gt. thresh .or. abs(lumo(n1)) .gt. thresh) then
              
             x1 = i + nint((trans_origin(1) + (Ni-1)*transition(1))/stepx)    ! always =i
             y1 = j + nint((trans_origin(2) + (Ni-1)*transition(2))/stepx)    ! always =j
             z1 = k + nint((trans_origin(3) + (Ni-1)*transition(3))/stepx)

              if (x1 .ge. 1 .and. x1 .le. nx) then
              if (y1 .ge. 1 .and. y1 .le. ny) then 
              if (z1 .ge. 1 .and. z1 .le. nz) then
                  n2 = (x1-1)*ny*nz+(y1-1)*nz+z1
                  overlaph(Ni) = overlaph(Ni) + homo(n1)*homo(n2)
                  overlapl(Ni) = overlapl(Ni) + lumo(n1)*lumo(n2)
                  ncount = ncount + 1
              end if
              end if
              end if

             end if

           end do
         end do
       end do
        write(*,*) "For cycle ", Ni, "number of overlap points:", ncount
        write(444,*) (Ni-1)*maxval(transition(:))*au2angstrom, (overlaph(Ni)), &
                      &      (overlapl(Ni)),(overlaph(Ni)*overlapl(Ni))
        write(555,*) (Ni-1)*maxval(transition(:))*au2angstrom, abs(overlaph(Ni)), & 
                      &      abs(overlapl(Ni)),abs(overlaph(Ni)*overlapl(Ni))           
  end do


end 

