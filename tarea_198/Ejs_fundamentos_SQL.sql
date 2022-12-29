--EJERCICIO 1
SELECT*FROM category;
Select name From category;
select name as genero FROM category;
--EJERCICIO 2
select distinct (first_name) as numero_filas From actor;
--EJERCICIO 3
SELECT*FROM film;
SELECT 
title as titulo, 
rental_duration as duracion_del_alquiler,
rental_rate as precio_alquiler
from film where film_id=5;
--Ejercicio 4
select*from film order by length asc;
--Ejercicio 5
select*from film where length<50 and  rental_rate=4.99;
--EJERCICIO 6
select * from payment
where payment_date between '2007-04-10' and '2007-04-18'
order by amount desc;
--EJERCICIO 7
select * from payment
where payment_date between '2007-04-10' and '2007-04-18'
and staff_id=2 and amount>7;
--EJERCICIO 8
select count (*)from payment where payment_date between '2007-04-10' and '2007-04-18'
and staff_id=2 and amount>7 