import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { RouterModule } from '@angular/router';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ContactoComponent } from './contacto/contacto.component';
import { PaginaPrincipalComponent } from './pagina-principal/pagina-principal.component';
import { ArticulosComponent } from './articulos/articulos.component';
import { QueEsMLComponent } from './Articulos/que-es-ml/que-es-ml.component';
import { MetodologiaCRISPComponent } from './Articulos/metodologia-crisp/metodologia-crisp.component';
import { HerramientasComponent } from './Articulos/herramientas/herramientas.component';
import { ComparacionComponent } from './Articulos/Herramientas/comparacion/comparacion.component';
import { TratamientoPrevioDeDatosComponent } from './Articulos/tratamiento-previo-de-datos/tratamiento-previo-de-datos.component';
import { EntrenamientoDeModelosComponent } from './Articulos/entrenamiento-de-modelos/entrenamiento-de-modelos.component';
import { PequenosProyectosComponent } from './pequenos-proyectos/pequenos-proyectos.component';
import { TitanicDatasetComponent } from './pequenos-proyectos/titanic-dataset/titanic-dataset.component';
import { WineDatasetComponent } from './pequenos-proyectos/wine-dataset/wine-dataset.component';
import { SportsDatasetComponent } from './pequenos-proyectos/sports-dataset/sports-dataset.component';

@NgModule({
  declarations: [
    AppComponent,
    ContactoComponent,
    PaginaPrincipalComponent,
    ArticulosComponent,
    QueEsMLComponent,
    MetodologiaCRISPComponent,
    HerramientasComponent,
    ComparacionComponent,
    TratamientoPrevioDeDatosComponent,
    EntrenamientoDeModelosComponent,
    PequenosProyectosComponent,
    TitanicDatasetComponent,
    WineDatasetComponent,
    SportsDatasetComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    RouterModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
