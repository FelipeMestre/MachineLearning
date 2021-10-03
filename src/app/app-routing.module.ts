import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AppComponent } from './app.component';
import { ArticulosComponent } from './articulos/articulos.component';
import { EntrenamientoDeModelosComponent } from './Articulos/entrenamiento-de-modelos/entrenamiento-de-modelos.component';
import { ComparacionComponent } from './Articulos/Herramientas/comparacion/comparacion.component';
import { HerramientasComponent } from './Articulos/herramientas/herramientas.component';
import { MetodologiaCRISPComponent } from './Articulos/metodologia-crisp/metodologia-crisp.component';
import { QueEsMLComponent } from './Articulos/que-es-ml/que-es-ml.component';
import { TratamientoPrevioDeDatosComponent } from './Articulos/tratamiento-previo-de-datos/tratamiento-previo-de-datos.component';
import { ContactoComponent } from './contacto/contacto.component';
import { PaginaPrincipalComponent } from './pagina-principal/pagina-principal.component';
import { PequenosProyectosComponent } from './pequenos-proyectos/pequenos-proyectos.component';
import { SportsDatasetComponent } from './pequenos-proyectos/sports-dataset/sports-dataset.component';
import { TitanicDatasetComponent } from './pequenos-proyectos/titanic-dataset/titanic-dataset.component';
import { WineDatasetComponent } from './pequenos-proyectos/wine-dataset/wine-dataset.component';

const routes: Routes = [
  { path: '', redirectTo: 'PaginaPrincipal', pathMatch: 'full' },
  {path: 'PaginaPrincipal', component: PaginaPrincipalComponent },
  {path: 'Contacto', component: ContactoComponent },
  {path: 'Articulos', component: ArticulosComponent },
  {path: 'Articulos/QueEsMl', component: QueEsMLComponent },
  {path: 'Articulos/MetodologiaCrisp', component: MetodologiaCRISPComponent },
  {path: 'Articulos/Herramientas', component: HerramientasComponent },
  {path: 'Articulos/Herramientas/Comparacion', component: ComparacionComponent },
  {path: 'Articulos/TratamientoPrevioDeDatos', component: TratamientoPrevioDeDatosComponent },
  {path: 'Articulos/EntrenamientoDeModelos', component: EntrenamientoDeModelosComponent },
  {path: 'PequenosProyectos', component: PequenosProyectosComponent },
  {path: 'PequenosProyectos/WineDataset', component: WineDatasetComponent },
  {path: 'PequenosProyectos/TitanicDataset', component: TitanicDatasetComponent },
  {path: 'PequenosProyectos/SportsDatset', component: SportsDatasetComponent },
  
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
