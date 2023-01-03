--Ir poniendo cada ejercicio como comentario y ejecutarlos de 1 en 1
--Ejercicio 1
CREATE DATABASE drogueriaJuncal
with 
owner=postgres
Encoding='UTF8'
Connection limit=-1;
--Ejercicio 2
Create table producto (
	id_producto serial primary key,
	nombre_producto VARCHAR(80) not null,
	precio_unitario numeric not null
	constraint precio_positivo check (precio_unitario>0)
);
--Ejercicio 3
Create table descuento (
	id_producto integer not null references producto(id_producto),
	descuento boolean default (FALSE),
	cantidad_descuento numeric not null default(0.0)
);
--Ejercicio 4
Create table cliente(
	id_cleinte serial primary key,
	nombre varchar (50) not null,
	apellido varchar(50) not null,
	fecha_nacimiento date not null,
	direccion varchar(100) not null,
	telefono varchar(15) not null, 
	fecha_alta date not null,
	CONSTRAINT edad_minima check (fecha_nacimiento<=(current_date-interval'18' year)),
	CONSTRAINT chk_phone check (telefono ~* '^\(+[0-9]{2}\)[0-9]{9}$')
	);
--A continuacion he importado el archivo CSV

--Ejercicio 5
insert into producto(id_producto, nombre_producto, precio_unitario)
values(1, 'pasta dientes trinaca', 2.3),
	(2, 'pasta de dientes mojate',3.00),
	(3, 'licor caribe', 4.65),
	(4, 'cepillo de dientes bambu', 1.00),
	(5, 'cepillo de diente juncal', 0.89);
--Ejercicio 6

Update producto
SET precio_unitario=0.99
where nombre_producto='cepillo de diente juncal';

--Ejercicio 7
delete from cliente
where (nombre like 'Borja' and apellido like 'Fernendez');
--Ejercicio 8
--No, para cambiar el nombre de varias columnas hay que ejecutar la sentencia de "alter table rename column repetidas veces"
--Ejercicio 9
ALTER TABLE cliente 
	rename column apellido to primer_apellido;
ALTER TABLE cliente 	
	rename column telefono to telefono_móvil;