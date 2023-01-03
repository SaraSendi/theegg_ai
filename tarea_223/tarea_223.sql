/* A continuación se agrupan diferentes ejercicios para aprender a trabajar con vistas*/
/*Cada ejercicio debe generarse de manera individual, si se intenta ejecutar varias veces el ejercicio 1 da error porque ya está generado, por ello aunque esté todo el código en este script debe ir comentándose para ejecutar cada ejercicio de manera individual*/
--Ejercicio 1: 
--Crea una vista con el nombre actores_peliculas_genero que muestre el nombre, y apellido de cada actor, el título de cada película en la que ha participado y el genero de esta.
CREATE VIEW actores_peliculas_genero AS 
select a.first_name as nombre_actor, 
a.last_name as apellido_actor,
f.title as titulo,
c.name as genero
from film f
inner join film_actor fa using(film_id) 
inner join actor a using(actor_id) 
inner join film_category fc using(film_id) 
inner join category c using(category_id);
--Ejercicio 2: 
--Consigue identificar todos los géneros en los que ha interpretado la actriz Penelope Guiness
select distinct (genero) as genero from actores_peliculas_genero
where nombre_actor='Penelope' and apellido_actor='Guiness';
--Ejercicio 3:
--Selecciona todas las películas en las que ha participado Jennifer Davis y que empiecen por la letra ‘B’ mayúscula.
select distinct (titulo) as tituloB from actores_peliculas_genero
where nombre_actor='Jennifer' and apellido_actor='Davis' and titulo like ('B%');
--Ejercicio 4: 
--Renombra la vista del Ejercicio 1 anterior a actor_film_genre
Alter view actores_peliculas_genero rename to actor_film_genre;*/
--Ejercicio 5: 
--Modifica la vista de actor_film_genre para introducir la duración de las películas como atributo duración
CREATE OR REPLACE VIEW actor_film_genre AS 
select a.first_name as nombre_actor, 
a.last_name as apellido_actor,
f.title as titulo,
c.name as genero,
f.length as duracion
from film f
inner join film_actor fa using(film_id) 
inner join actor a using(actor_id) 
inner join film_category fc using(film_id) 
inner join category c using(category_id);