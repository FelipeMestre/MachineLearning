import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PequenosProyectosComponent } from './pequenos-proyectos.component';

describe('PequenosProyectosComponent', () => {
  let component: PequenosProyectosComponent;
  let fixture: ComponentFixture<PequenosProyectosComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ PequenosProyectosComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(PequenosProyectosComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
